# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Shapes
#
# Shows how to programmatically creates a variety of
# collision shapes using the newton.ModelBuilder() API.
#
# Command: python -m newton.examples basic_shapes
#
###########################################################################

import warp as wp
import numpy as np
from pxr import Usd

import newton
import newton.examples
import newton.usd

from newton._src.solvers.xpbd.new_solver_xpbd import NewSolverXPBD


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # z height to drop shapes from
        drop_z = 2.0

        # Store body refs
        self.body_refs = {}

        # SPHERE
        self.sphere_pos = wp.vec3(0.0, 2.0, drop_z)
        body_sphere = builder.add_body(xform=wp.transform(p=self.sphere_pos, q=wp.quat_identity()), key="sphere")
        builder.add_shape_sphere(body_sphere, radius=0.5)
        self.body_refs["sphere"] = body_sphere

        # CAPSULE
        self.capsule_pos = wp.vec3(0.0, 6.0, drop_z+2)
        body_capsule = builder.add_body(xform=wp.transform(p=self.capsule_pos, q=wp.quat_identity()), key="capsule")
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)
        self.body_refs["capsule"] = body_capsule

        # CYLINDER
        self.cylinder_pos = wp.vec3(0.0, 4.0, drop_z+4)
        body_cylinder = builder.add_body(xform=wp.transform(p=self.cylinder_pos, q=wp.quat_identity()), key="cylinder")
        builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)
        self.body_refs["cylinder"] = body_cylinder

        # BOX
        self.box_pos = wp.vec3(0.0, 6.0, drop_z+6)
        body_box = builder.add_body(xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), key="box")
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)
        self.body_refs["box"] = body_box

        # MESH (bunny)
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        self.mesh_pos = wp.vec3(0.0, 8.0, drop_z + 8)
        body_mesh = builder.add_body(xform=wp.transform(p=self.mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), key="mesh")
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)
        self.body_refs["mesh"] = body_mesh


        # CONE (no collision support in the standard collision pipeline)
        self.cone_pos = wp.vec3(0.0, 10.0, drop_z+10)
        body_cone = builder.add_body(xform=wp.transform(p=self.cone_pos, q=wp.quat_identity()), key="cone")
        builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)
        self.body_refs["cone"] = body_cone

        # SOFT BODY (a deformable cube)
        self.soft_cube_pos = wp.vec3(0.0, 8.0, drop_z)

        # Create a simple 3x3x3 grid of particles
        cube_particles = []
        grid_size = 3
        spacing = 0.8

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if (i == 1 and j == 1 and k == 1):
                        pos = self.soft_cube_pos + wp.vec3(i * spacing, j * spacing, k * spacing)
                        p = builder.add_particle(pos, vel=wp.vec3(0.0, 0.0, 0.0), mass=0.0, radius=0.2)
                        cube_particles.append(p)
                    else:
                        pos = self.soft_cube_pos + wp.vec3(i * spacing, j * spacing, k * spacing)
                        p = builder.add_particle(pos, vel=wp.vec3(0.0, 0.0, 0.0), mass=10.0, radius=0.2)
                        cube_particles.append(p)

        # Add springs between neighboring particles
        def get_index(i, j, k):
            return i * grid_size * grid_size + j * grid_size + k

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    idx = get_index(i, j, k)
                    # Connect to neighbors
                    if i < grid_size - 1:
                        builder.add_spring(cube_particles[idx], cube_particles[get_index(i+1, j, k)], 
                                         ke=1000000, kd=100, control=0.0)
                    if j < grid_size - 1:
                        builder.add_spring(cube_particles[idx], cube_particles[get_index(i, j+1, k)], 
                                         ke=1000000, kd=100, control=0.0)
                    if k < grid_size - 1:
                        builder.add_spring(cube_particles[idx], cube_particles[get_index(i, j, k+1)], 
                                         ke=1000000, kd=100, control=0.0)

        
        # finalize model
        self.model = builder.finalize()

        self.body_names = {v: k for k, v in self.body_refs.items()}

        self.solver = NewSolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create collision pipeline from command-line args (default: CollisionPipelineUnified with EXPLICIT)
        # Override rigid_contact_max_per_pair because mesh vs plane creates a lot of contacts
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            rigid_contact_max_per_pair=100,
            rigid_contact_margin=0.05,
        )
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)


            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        if int(self.sim_time * self.fps) % 15 != 0:
            return

         # accessing lambda
        if self.solver.last_spring_lambdas is not None:
            lam = self.solver.last_spring_lambdas.numpy()
        if self.solver.last_edge_lambdas is not None:
            lam = self.solver.last_edge_lambdas.numpy()
            print("Edge lambdas:", lam[:10])

        if self.solver.last_contact_count is not None:
            contact_count = self.solver.last_contact_count.numpy()[0]
        
        if contact_count > 0:
            print(f"\n=== Frame {int(self.sim_time * self.fps)} ===")
            print(f"Active contacts: {contact_count}")
            
            # Get contact information
            # normals = self.solver.last_contact_normals.numpy()
            shape0 = self.solver.last_contact_shapes[0].numpy()
            shape1 = self.solver.last_contact_shapes[1].numpy()
            forces = self.solver.last_contact_forces.numpy()

            from collections import defaultdict
            body_pair_contacts = defaultdict(int)
            body_pair_forces = defaultdict(list)
            
            # Analyze each contact
            for c in range(contact_count):  # Show first 5 contacts
                # normal = normals[c]
                s0 = shape0[c]
                s1 = shape1[c]
                
                # Get body indices from shapes
                body0 = self.model.shape_body.numpy()[s0]
                body1 = self.model.shape_body.numpy()[s1]

                # body0_name = self.body_names.get(body0, f"Body_{body0}") if body0 >= 0 else "Ground"
                # body1_name = self.body_names.get(body1, f"Body_{body1}") if body1 >= 0 else "Ground"
                
                # # Calculate force magnitude from body_deltas
                # # body_deltas contains velocity changes, convert to force
                # force_body0 = forces[body0] if body0 >= 0 else [0,0,0,0,0,0]
                # force_body1 = forces[body1] if body1 >= 0 else [0,0,0,0,0,0]
                
                # print(f"Contact {c}:")
                # print(f"  {body0_name} {body0} <-> {body1_name} {body1}")
                # print(f"  Normal: {normal}")
                # print(f"  Force on body {body0_name}: {force_body0[:3]}")  # Linear component
                # print(f"  Force on body {body1_name}: {force_body1[:3]}")
                pair = tuple(sorted([body0, body1]))
                body_pair_contacts[pair] += 1

                forces - forces[body0] if body0 >= 0 else forces[body1]
                body_pair_forces[pair].append(forces[:3])
            print("\nContacts by body pair:")
            for pair, count in sorted(body_pair_contacts.items()):
                body0_name = self.body_names.get(pair[0], f"Body_{pair[0]}") if pair[0] >= 0 else "Ground"
                body1_name = self.body_names.get(pair[1], f"Body_{pair[1]}") if pair[1] >= 0 else "Ground"
                
                # Average force across all contact points
                avg_force = np.mean(body_pair_forces[pair], axis=0)
                avg_force = np.round(avg_force, 3)

                print(f"  {body0_name} <-> {body1_name}: {count} contact points, avg force: {avg_force}")

    def test(self):
        self.sphere_pos[2] = 0.5
        sphere_q = wp.transform(self.sphere_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "sphere at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, sphere_q, atol=2e-4),
            [0],
        )
        self.capsule_pos[2] = 1.0
        capsule_q = wp.transform(self.capsule_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "capsule at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, capsule_q, atol=2e-4),
            [1],
        )
        # Custom test for cylinder: allow 0.01 error for X and Y, strict for Z and rotation
        self.cylinder_pos[2] = 0.6
        cylinder_q = wp.transform(self.cylinder_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cylinder at rest pose",
            lambda q, qd: abs(q[0] - cylinder_q[0]) < 0.01
            and abs(q[1] - cylinder_q[1]) < 0.01
            and abs(q[2] - cylinder_q[2]) < 1e-4
            and abs(q[3] - cylinder_q[3]) < 1e-4
            and abs(q[4] - cylinder_q[4]) < 1e-4
            and abs(q[5] - cylinder_q[5]) < 1e-4
            and abs(q[6] - cylinder_q[6]) < 1e-4,
            [2],
        )
        self.box_pos[2] = 0.25
        box_q = wp.transform(self.box_pos, wp.quat_identity())
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "box at rest pose",
            lambda q, qd: newton.utils.vec_allclose(q, box_q, atol=0.1),
            [3],
        )
        # we only test that the bunny didn't fall through the ground and didn't slide too far
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "bunny at rest pose",
            lambda q, qd: q[2] > 0.01 and abs(q[0]) < 0.1 and abs(q[1] - 4.0) < 0.1,
            [4],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer, args)

    # print("UPDATED FILE")

    newton.examples.run(example, args)
