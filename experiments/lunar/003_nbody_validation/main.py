"""Idealized Earth/Moon + passive spacecraft, SI inertial barycentric state."""
import numpy as np
from orbit_sim.dynamics import G, M_EARTH
from orbit_sim.integrators import Velocity_Verlet_KDK, Velocity_Verlet_DKD
from orbit_sim.simulate import simulate_nbody


def initial_state(spacecraft=True):
    masses = np.array([M_EARTH, 7.342e22])
    distance = 3.844e8
    fractions = masses / masses.sum()
    speed = np.sqrt(G * masses.sum() / distance)
    positions = np.array([[-fractions[1]*distance, 0., 0.],
                          [fractions[0]*distance, 0., 0.]])
    velocities = np.array([[0., -fractions[1]*speed, 0.],
                           [0., fractions[0]*speed, 0.]])
    mask = np.array([True, True])
    if spacecraft:
        positions = np.vstack([positions, positions[0] + [1.e7, 0., 0.]])
        velocities = np.vstack([velocities, velocities[0] + [0., np.sqrt(G*M_EARTH/1.e7), 0.]])
        masses = np.append(masses, 1000.)
        mask = np.append(mask, False)
    period = 2*np.pi*distance/speed
    return positions, velocities, masses, mask, period


def conservation_errors(data, r, v, m):
    energy = np.max(np.abs(data['total_energy']/data['total_energy'][0] - 1))
    angular = np.max(np.linalg.norm(data['total_angular_momentum'] - data['total_angular_momentum'][0], axis=1)) / np.linalg.norm(data['total_angular_momentum'][0])
    # Initial total momentum is nearly zero: normalize by sum |m_i v_i|.
    scale = np.sum(m * np.linalg.norm(v, axis=1))
    momentum = np.max(np.linalg.norm(data['total_linear_momentum'] - data['total_linear_momentum'][0], axis=1))/scale
    return energy, angular, momentum


def main():
    r, v, m, mask, period = initial_state(False)
    for method in (Velocity_Verlet_KDK, Velocity_Verlet_DKD):
        result = simulate_nbody(r, v, m, period/400, 8000, method, mask)
        errors = conservation_errors(result, r, v, m)
        assert errors[0] < 1.e-6 and max(errors[1:]) < 1.e-11
        coarse = simulate_nbody(r, v, m, period/200, 200, method, mask)
        fine = simulate_nbody(r, v, m, period/400, 400, method, mask)
        ratio = np.linalg.norm(coarse['positions'][-1]-r)/np.linalg.norm(fine['positions'][-1]-r)
        assert 3.8 < ratio < 4.2
        print(f'{method.__name__}: 20 periods PASS; E/L/P errors={errors}; position convergence ratio={ratio:.4f}')
    r, v, m, mask, period = initial_state()
    result = simulate_nbody(r, v, m, 20., 5000, Velocity_Verlet_KDK, mask)
    active = simulate_nbody(r[:2], v[:2], m[:2], 20., 5000, Velocity_Verlet_KDK)
    np.testing.assert_array_equal(result['positions'][:, :2], active['positions'])
    assert np.all(np.isfinite(result['positions']))
    displacement = np.linalg.norm(result['positions'][-1, 2]-r[2])
    assert displacement > 0
    print(f'Earth/Moon + test spacecraft: PASS; active trajectories identical; spacecraft displacement={displacement:.3e} m')


if __name__ == '__main__':
    main()
