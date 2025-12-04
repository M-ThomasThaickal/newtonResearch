# from newton.solvers import SolverXPBD
# from newton._src.core.types import override
# from newton._src.sim import Contacts, Control, State
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from .solver_xpbd import SolverXPBD
from .kernels import (
    apply_body_delta_velocities,
    apply_body_deltas,
    apply_joint_forces,
    apply_particle_deltas,
    apply_particle_shape_restitution,
    apply_rigid_restitution,
    bending_constraint,
    solve_body_contact_positions,
    solve_body_joints,
    solve_particle_particle_contacts,
    solve_particle_shape_contacts,
    # solve_simple_body_joints,
    solve_springs,
    solve_tetrahedra,
    update_body_velocities,
)


# make a copy of kernels file and output lambda in each kernel that uses lambda the same way the output they
# already have are outputted. Then go to this file, overwrite the functions from solver_xpbd into this solver
# which inherits solver_xpbd that use wp.launch a kernel that has been modified with the lambda output
# add that as an output.

class NewSolverXPBD(SolverXPBD):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_spring_lambdas = None
        self.last_edge_lambdas = None

        self.last_contact_count = None
        self.last_contact_normals = None
        self.last_contact_shapes = None
        self.last_contact_forces = None


    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        model = self.model

        particle_q = None
        particle_qd = None
        particle_deltas = None

        body_q = None
        body_qd = None
        body_deltas = None

        rigid_contact_inv_weight = None

        if contacts:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros_like(contacts.rigid_contact_thickness0)
            rigid_contact_inv_weight_init = None

        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd

                self.particle_q_init = wp.clone(state_in.particle_q)
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_in.particle_qd)
                particle_deltas = wp.empty_like(state_out.particle_qd)

                self.integrate_particles(model, state_in, state_out, dt)

            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)

                body_deltas = wp.empty_like(state_out.body_qd)

                if model.joint_count:
                    wp.launch(
                        kernel=apply_joint_forces,
                        dim=model.joint_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_axis,
                            control.joint_f,
                        ],
                        outputs=[state_in.body_f],
                        device=model.device,
                    )

                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

            spring_constraint_lambdas = None
            if model.spring_count:
                spring_constraint_lambdas = wp.empty_like(model.spring_rest_length)
            edge_constraint_lambdas = None
            if model.edge_count:
                edge_constraint_lambdas = wp.empty_like(model.edge_rest_angle)

            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.body_count:
                        if requires_grad and i > 0:
                            body_deltas = wp.zeros_like(body_deltas)
                        else:
                            body_deltas.zero_()

                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas)
                        else:
                            particle_deltas.zero_()

                        # particle-rigid body contacts (besides ground plane)
                        if model.shape_count:
                            wp.launch(
                                kernel=solve_particle_shape_contacts, # change kernel
                                dim=contacts.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    model.body_inv_mass,
                                    model.body_inv_inertia,
                                    model.shape_body,
                                    model.shape_material_mu,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    contacts.soft_contact_count,
                                    contacts.soft_contact_particle,
                                    contacts.soft_contact_shape,
                                    contacts.soft_contact_body_pos,
                                    contacts.soft_contact_body_vel,
                                    contacts.soft_contact_normal,
                                    contacts.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                # outputs
                                outputs=[particle_deltas, body_deltas], # add output
                                device=model.device, 
                            )

                        if model.particle_max_radius > 0.0 and model.particle_count > 1:
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id,
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
                            spring_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=solve_springs,
                                dim=model.spring_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.spring_indices,
                                    model.spring_rest_length,
                                    model.spring_stiffness,
                                    model.spring_damping,
                                    dt,
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )
                            if i == self.iterations - 1:
                                self.last_spring_lambdas = spring_constraint_lambdas

                        # bending constraints
                        if model.edge_count:
                            edge_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_bending_properties,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )
                            if i == self.iterations - 1:
                                self.last_edge_lambdas = edge_constraint_lambdas

                        # tetrahedral FEM
                        if model.tet_count:
                            wp.launch(
                                kernel=solve_tetrahedra,
                                dim=model.tet_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tet_indices,
                                    model.tet_poses,
                                    model.tet_activations,
                                    model.tet_materials,
                                    dt,
                                    self.soft_body_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        particle_q, particle_qd = self.apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                    # handle rigid bodies
                    # ----------------------------

                    if model.joint_count:
                        # wp.launch(
                        #     kernel=solve_simple_body_joints,
                        #     dim=model.joint_count,
                        #     inputs=[
                        #         body_q,
                        #         body_qd,
                        #         model.body_com,
                        #         model.body_inv_mass,
                        #         model.body_inv_inertia,
                        #         model.joint_type,
                        #         model.joint_enabled,
                        #         model.joint_parent,
                        #         model.joint_child,
                        #         model.joint_X_p,
                        #         model.joint_X_c,
                        #         model.joint_limit_lower,
                        #         model.joint_limit_upper,
                        #         model.joint_qd_start,
                        #         model.joint_dof_dim,
                        #         model.joint_dof_mode,
                        #         model.joint_axis,
                        #         control.joint_target,
                        #         model.joint_target_ke,
                        #         model.joint_target_kd,
                        #         self.joint_linear_compliance,
                        #         self.joint_angular_compliance,
                        #         self.joint_angular_relaxation,
                        #         self.joint_linear_relaxation,
                        #         dt,
                        #     ],
                        #     outputs=[body_deltas],
                        #     device=model.device,
                        # )

                        wp.launch(
                            kernel=solve_body_joints,
                            dim=model.joint_count,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.joint_type,
                                model.joint_enabled,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_X_p,
                                model.joint_X_c,
                                model.joint_limit_lower,
                                model.joint_limit_upper,
                                model.joint_qd_start,
                                model.joint_dof_dim,
                                model.joint_axis,
                                control.joint_target_pos,
                                control.joint_target_vel,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                self.joint_linear_compliance,
                                self.joint_angular_compliance,
                                self.joint_angular_relaxation,
                                self.joint_linear_relaxation,
                                dt,
                            ],
                            outputs=[body_deltas],
                            device=model.device,
                        )

                        body_q, body_qd = self.apply_body_deltas(model, state_in, state_out, body_deltas, dt)

                    # Solve rigid contact constraints
                    if model.body_count and contacts is not None:
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight.zero_()
                        body_deltas.zero_()

                        wp.launch(
                            kernel=solve_body_contact_positions,
                            dim=contacts.rigid_contact_max,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.shape_body,
                                contacts.rigid_contact_count,
                                contacts.rigid_contact_point0,
                                contacts.rigid_contact_point1,
                                contacts.rigid_contact_offset0,
                                contacts.rigid_contact_offset1,
                                contacts.rigid_contact_normal,
                                contacts.rigid_contact_thickness0,
                                contacts.rigid_contact_thickness1,
                                contacts.rigid_contact_shape0,
                                contacts.rigid_contact_shape1,
                                model.shape_material_mu,
                                self.rigid_contact_relaxation,
                                dt,
                                model.rigid_contact_torsional_friction,
                                model.rigid_contact_rolling_friction,
                            ],
                            outputs=[
                                body_deltas,
                                rigid_contact_inv_weight,
                            ],
                            device=model.device,
                        )

                        if i == self.iterations - 1:
                            self.last_contact_count = contacts.rigid_contact_count
                            self.last_contact_normals = contacts.rigid_contact_normal
                            self.last_contact_shapes = (contacts.rigid_contact_shape0, contacts.rigid_contact_shape1)
                            self.last_contact_forces = body_deltas 
                        # if model.rigid_contact_count.numpy()[0] > 0:
                        #     print("rigid_contact_count:", model.rigid_contact_count.numpy().flatten())
                        #     # print("rigid_active_contact_distance:", rigid_active_contact_distance.numpy().flatten())
                        #     # print("rigid_active_contact_point0:", rigid_active_contact_point0.numpy().flatten())
                        #     # print("rigid_active_contact_point1:", rigid_active_contact_point1.numpy().flatten())
                        #     print("body_deltas:", body_deltas.numpy().flatten())

                        # print(rigid_active_contact_distance.numpy().flatten())

                        if self.enable_restitution and i == 0:
                            # remember contact constraint weighting from the first iteration
                            if self.rigid_contact_con_weighting:
                                rigid_contact_inv_weight_init = wp.clone(rigid_contact_inv_weight)
                            else:
                                rigid_contact_inv_weight_init = None

                        body_q, body_qd = self.apply_body_deltas(
                            model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                        )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)

            # update body velocities from position changes
            if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                # causes gradient issues (probably due to numerical problems
                # when computing velocities from position changes)
                if requires_grad:
                    out_body_qd = wp.clone(state_out.body_qd)
                else:
                    out_body_qd = state_out.body_qd

                # update body velocities
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

            if self.enable_restitution and contacts is not None:
                if model.particle_count:
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=model.particle_count,
                        inputs=[
                            particle_q,
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            body_q,
                            body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
                            contacts.soft_contact_count,
                            contacts.soft_contact_particle,
                            contacts.soft_contact_shape,
                            contacts.soft_contact_body_pos,
                            contacts.soft_contact_body_vel,
                            contacts.soft_contact_normal,
                            contacts.soft_contact_max,
                            dt,
                            self.soft_contact_relaxation,
                        ],
                        outputs=[state_out.particle_qd],
                        device=model.device,
                    )

                if model.body_count:
                    body_deltas.zero_()

                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            body_q_init,
                            body_qd_init,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            model.shape_material_restitution,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_offset0,
                            contacts.rigid_contact_offset1,
                            contacts.rigid_contact_thickness0,
                            contacts.rigid_contact_thickness1,
                            rigid_contact_inv_weight_init,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            return state_out
