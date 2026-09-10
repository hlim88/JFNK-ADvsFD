"""Linear Su--Olson diffusion."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from trt_jfnk.discretization.boundary import BoundaryCondition2D, enforce_radiation_residual
from trt_jfnk.discretization.diffusion import laplacian
from trt_jfnk.discretization.grid import Grid2D
from trt_jfnk.discretization.time_integrators import ThetaMethod


@dataclass(frozen=True)
class SuOlsonLinearModel:
    r"""Linear reference equations.

    .. math::
       U_t=\tfrac13\nabla^2 U-U+V+Q,\qquad
       V_t=\epsilon(U-V).
    """

    epsilon: float = 1.0e-2
    diffusion: float = 1.0 / 3.0
    integrator: ThetaMethod = field(default_factory=lambda: ThetaMethod(0.5))

    def pack(self, radiation, material):
        return jnp.concatenate((radiation.reshape(-1), material.reshape(-1)))

    def unpack(self, state, grid: Grid2D):
        if state.size != 2 * grid.size:
            raise ValueError("state must contain two grid-sized blocks")
        return state[: grid.size].reshape(grid.shape), state[grid.size :].reshape(grid.shape)

    def rhs(self, radiation, material, source, grid: Grid2D, bc: BoundaryCondition2D):
        radiation_rhs = self.diffusion * laplacian(radiation, grid, bc) - radiation + material + source
        material_rhs = self.epsilon * (radiation - material)
        return radiation_rhs, material_rhs

    def residual(self, state, old_state, dt, source_new, source_old, grid, bc):
        radiation, material = self.unpack(state, grid)
        radiation_old, material_old = self.unpack(old_state, grid)
        rhs_r, rhs_m = self.rhs(radiation, material, source_new, grid, bc)
        rhs_r_old, rhs_m_old = self.rhs(radiation_old, material_old, source_old, grid, bc)
        residual_r = radiation - radiation_old - dt * self.integrator.blend(rhs_r, rhs_r_old)
        residual_m = material - material_old - dt * self.integrator.blend(rhs_m, rhs_m_old)
        residual_r = enforce_radiation_residual(residual_r, radiation, grid, bc)
        return self.pack(residual_r, residual_m)

    def is_admissible(self, state, grid, radiation_floor=0.0, material_floor=0.0):
        radiation, material = self.unpack(state, grid)
        return jnp.logical_and(
            jnp.all(radiation >= radiation_floor), jnp.all(material >= material_floor)
        )
