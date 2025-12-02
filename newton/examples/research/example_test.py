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
# Example XPBD Cubes with Contact Forces
#
# Builds a minimal XPBD scene with two dynamic cubes above a plane. The
# viewer supports ray casting and mouse picking (right-click drag) through
# the default picking handler, and when the cubes collide the reaction
# forces inferred from their momentum change are printed to the terminal.
#
# Command: python -m newton.examples xpbd_cubes
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        # setup simulation parameters first
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # create two cubes and a plane
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.6
        builder.default_shape_cfg.restitution = 0.05

        cube_half_extent = 0.25
        cube_start_height = 1.0

        self.cube_a_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.35, 0.0, cube_start_height + 0.1), q=wp.quat_identity()),
            key="cube_a",
        )
        self.cube_a_shape = builder.add_shape_box(self.cube_a_body, hx=cube_half_extent, hy=cube_half_extent, hz=cube_half_extent)

        self.cube_b_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.35, 0.0, cube_start_height), q=wp.quat_identity()),
            key="cube_b",
        )
        self.cube_b_shape = builder.add_shape_box(self.cube_b_body, hx=cube_half_extent, hy=cube_half_extent, hz=cube_half_extent)

        builder.add_ground_plane()

        # finalize model and solver
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=12)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # not required for MuJoCo, but required for maximal-coordinate solvers like XPBD
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

        # cache mass and gravity for reaction force calculation
        self.body_mass = np.array(self.model.body_mass.numpy())
        self.gravity = np.array(self.model.gravity.numpy()[0])

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _linear_velocities(self, body_qd: wp.array) -> np.ndarray:
        qd_np = np.array(body_qd.numpy(), copy=True)
        if qd_np.ndim == 1:
            qd_np = qd_np.reshape((-1, 6))
        return qd_np[:, :3]

    def _report_reaction_forces(self, pre_vel: np.ndarray, post_vel: np.ndarray, contact_count: int) -> None:
        if contact_count <= 0:
            return

        shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]

        in_contact = False
        for s0, s1 in zip(shape0, shape1):
            pair = {s0, s1}
            if self.cube_a_shape in pair and self.cube_b_shape in pair:
                in_contact = True
                break

        if not in_contact:
            return

        impulse_a = self.body_mass[self.cube_a_body] * (post_vel[self.cube_a_body] - pre_vel[self.cube_a_body] - self.sim_dt * self.gravity)
        impulse_b = self.body_mass[self.cube_b_body] * (post_vel[self.cube_b_body] - pre_vel[self.cube_b_body] - self.sim_dt * self.gravity)

        reaction_a = impulse_a / self.sim_dt
        reaction_b = impulse_b / self.sim_dt

        print(
            "Cube contact reaction forces (world frame):\n"
            f"  cube_a: {reaction_a} N\n"
            f"  cube_b: {reaction_b} N"
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            pre_vel = self._linear_velocities(self.state_0.body_qd)

            self.state_0.clear_forces()

            # apply forces to the model (includes picking and wind from the viewer)
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            contact_count = int(self.contacts.rigid_contact_count.numpy()[0])

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            post_vel = self._linear_velocities(self.state_1.body_qd)
            self._report_reaction_forces(pre_vel, post_vel, contact_count)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt


if __name__ == "__main__":
    newton.examples.run(Example)