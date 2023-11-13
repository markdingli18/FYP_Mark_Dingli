import gc
import math
from argparse import ArgumentParser
from datetime import timedelta as delta

import numpy as np
import pytest

from parcels import (
    AdvectionEE,
    AdvectionRK4,
    AdvectionRK45,
    FieldSet,
    JITParticle,
    ParticleSet,
    ScipyParticle,
    download_example_dataset,
)

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}


def moving_eddies_fieldset(xdim=200, ydim=350, mesh='flat'):
    """Generate a fieldset encapsulating the flow field consisting of two
    moving eddies, one moving westward and the other moving northwestward.

    Parameters
    ----------
    xdim :
        Horizontal dimension of the generated fieldset (Default value = 200)
    xdim :
        Vertical dimension of the generated fieldset (Default value = 200)
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical: Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat  (default): No conversion, lat/lon are assumed to be in m.
    ydim :
         (Default value = 350)


    Notes
    -----
    Note that this is not a proper geophysical flow. Rather, a Gaussian eddy is moved
    artificially with uniform velocities. Velocities are calculated from geostrophy.

    """
    # Set Parcels FieldSet variables
    time = np.arange(0., 8. * 86400., 86400., dtype=np.float64)

    # Coordinates of the test fieldset (on A-grid in m)
    if mesh == 'spherical':
        lon = np.linspace(0, 4, xdim, dtype=np.float32)
        lat = np.linspace(45, 52, ydim, dtype=np.float32)
    else:
        lon = np.linspace(0, 4.e5, xdim, dtype=np.float32)
        lat = np.linspace(0, 7.e5, ydim, dtype=np.float32)

    # Grid spacing in m
    def cosd(x):
        return math.cos(math.radians(float(x)))
    dx = (lon[1] - lon[0]) * 1852 * 60 * cosd(lat.mean()) if mesh == 'spherical' else lon[1] - lon[0]
    dy = (lat[1] - lat[0]) * 1852 * 60 if mesh == 'spherical' else lat[1] - lat[0]

    # Define arrays U (zonal), V (meridional), and P (sea surface height) on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    # Some constants
    corio_0 = 1.e-4  # Coriolis parameter
    h0 = 1  # Max eddy height
    sig = 0.5  # Eddy e-folding decay scale (in degrees)
    g = 10  # Gravitational constant
    eddyspeed = 0.1  # Translational speed in m/s
    dX = eddyspeed * 86400 / dx  # Grid cell movement of eddy max each day
    dY = eddyspeed * 86400 / dy  # Grid cell movement of eddy max each day

    [x, y] = np.mgrid[:lon.size, :lat.size]
    for t in range(time.size):
        hymax_1 = lat.size / 7.
        hxmax_1 = .75 * lon.size - dX * t
        hymax_2 = 3. * lat.size / 7. + dY * t
        hxmax_2 = .75 * lon.size - dX * t

        P[:, :, t] = h0 * np.exp(-(x-hxmax_1)**2/(sig*lon.size/4.)**2-(y-hymax_1)**2/(sig*lat.size/7.)**2)
        P[:, :, t] += h0 * np.exp(-(x-hxmax_2)**2/(sig*lon.size/4.)**2-(y-hymax_2)**2/(sig*lat.size/7.)**2)

        V[:-1, :, t] = -np.diff(P[:, :, t], axis=0) / dx / corio_0 * g
        V[-1, :, t] = V[-2, :, t]  # Fill in the last column

        U[:, :-1, t] = np.diff(P[:, :, t], axis=1) / dy / corio_0 * g
        U[:, -1, t] = U[:, -2, t]  # Fill in the last row

    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lon': lon, 'lat': lat, 'time': time}
    return FieldSet.from_data(data, dimensions, transpose=True, mesh=mesh)


def moving_eddies_example(fieldset, outfile, npart=2, mode='jit', verbose=False,
                          method=AdvectionRK4):
    """Configuration of a particle set that follows two moving eddies.


    Parameters
    ----------
    fieldset :
        :class FieldSet: that defines the flow field
    outfile :

    npart :
         Number of particles to initialise. (Default value = 2)
    mode :
         (Default value = 'jit')
    verbose :
         (Default value = False)
    method :
         (Default value = AdvectionRK4)
    """
    # Determine particle class according to mode
    start = (3.3, 46.) if fieldset.U.grid.mesh == 'spherical' else (3.3e5, 1e5)
    finish = (3.3, 47.8) if fieldset.U.grid.mesh == 'spherical' else (3.3e5, 2.8e5)
    pset = ParticleSet.from_line(fieldset=fieldset, size=npart, pclass=ptype[mode],
                                 start=start, finish=finish)

    if verbose:
        print(f"Initial particle positions:\n{pset}")

    # Execute for 1 week, with 1 hour timesteps and hourly output
    runtime = delta(days=7)
    print("MovingEddies: Advecting %d particles for %s" % (npart, str(runtime)))
    pset.execute(method, runtime=runtime, dt=delta(hours=1),
                 output_file=pset.ParticleFile(name=outfile, outputdt=delta(hours=1)))

    if verbose:
        print(f"Final particle positions:\n{pset}")

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_moving_eddies_fwdbwd(mode, mesh, tmpdir, npart=2):
    method = AdvectionRK4
    fieldset = moving_eddies_fieldset(mesh=mesh)

    # Determine particle class according to mode
    lons = [3.3, 3.3] if fieldset.U.grid.mesh == 'spherical' else [3.3e5, 3.3e5]
    lats = [46., 47.8] if fieldset.U.grid.mesh == 'spherical' else [1e5, 2.8e5]
    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode], lon=lons, lat=lats)

    # Execte for 14 days, with 30sec timesteps and hourly output
    runtime = delta(days=1)
    dt = delta(minutes=5)
    outputdt = delta(hours=1)
    print("MovingEddies: Advecting %d particles for %s" % (npart, str(runtime)))
    outfile = tmpdir.join("EddyParticlefwd")
    pset.execute(method, runtime=runtime, dt=dt,
                 output_file=pset.ParticleFile(name=outfile, outputdt=outputdt))

    print("Now running in backward time mode")
    outfile = tmpdir.join("EddyParticlebwd")
    pset.execute(method, endtime=0, dt=-dt,
                 output_file=pset.ParticleFile(name=outfile, outputdt=outputdt))

    # Also include last timestep
    for var in ['lon', 'lat', 'depth', 'time']:
        pset.particledata.setallvardata(f"{var}", pset.particledata.getvardata(f"{var}_nextloop"))

    assert np.allclose(pset.lon, lons)
    assert np.allclose(pset.lat, lats)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_moving_eddies_fieldset(mode, mesh, tmpdir):
    fieldset = moving_eddies_fieldset(mesh=mesh)
    outfile = tmpdir.join("EddyParticle")
    pset = moving_eddies_example(fieldset, outfile, 2, mode=mode)
    # Also include last timestep
    for var in ['lon', 'lat', 'depth', 'time']:
        pset.particledata.setallvardata(f"{var}", pset.particledata.getvardata(f"{var}_nextloop"))
    if mesh == 'flat':
        assert (pset[0].lon < 2.2e5 and 1.1e5 < pset[0].lat < 1.2e5)
        assert (pset[1].lon < 2.2e5 and 3.7e5 < pset[1].lat < 3.8e5)
    else:
        assert (pset[0].lon < 2.0 and 46.2 < pset[0].lat < 46.25)
        assert (pset[1].lon < 2.0 and 48.8 < pset[1].lat < 48.85)


def fieldsetfile(mesh, tmpdir):
    """Generate fieldset files for moving_eddies test."""
    filename = tmpdir.join('moving_eddies')
    fieldset = moving_eddies_fieldset(200, 350, mesh=mesh)
    fieldset.write(filename)
    return filename


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_moving_eddies_file(mode, mesh, tmpdir):
    gc.collect()
    fieldset = FieldSet.from_parcels(fieldsetfile(mesh, tmpdir), extra_fields={'P': 'P'})
    outfile = tmpdir.join("EddyParticle")
    pset = moving_eddies_example(fieldset, outfile, 2, mode=mode)
    # Also include last timestep
    for var in ['lon', 'lat', 'depth', 'time']:
        pset.particledata.setallvardata(f"{var}", pset.particledata.getvardata(f"{var}_nextloop"))
    if mesh == 'flat':
        assert (pset[0].lon < 2.2e5 and 1.1e5 < pset[0].lat < 1.2e5)
        assert (pset[1].lon < 2.2e5 and 3.7e5 < pset[1].lat < 3.8e5)
    else:
        assert (pset[0].lon < 2.0 and 46.2 < pset[0].lat < 46.25)
        assert (pset[1].lon < 2.0 and 48.8 < pset[1].lat < 48.85)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_periodic_and_computeTimeChunk_eddies(mode):
    data_folder = download_example_dataset("MovingEddies_data")
    filename = str(data_folder / "moving_eddies")

    fieldset = FieldSet.from_parcels(filename)
    fieldset.add_constant('halo_west', fieldset.U.grid.lon[0])
    fieldset.add_constant('halo_east', fieldset.U.grid.lon[-1])
    fieldset.add_constant('halo_south', fieldset.U.grid.lat[0])
    fieldset.add_constant('halo_north', fieldset.U.grid.lat[-1])
    fieldset.add_periodic_halo(zonal=True, meridional=True)
    pset = ParticleSet.from_list(fieldset=fieldset,
                                 pclass=ptype[mode],
                                 lon=[3.3, 3.3],
                                 lat=[46.0, 47.8])

    def periodicBC(particle, fieldset, time):
        if particle.lon < fieldset.halo_west:
            particle_dlon += fieldset.halo_east - fieldset.halo_west  # noqa
        elif particle.lon > fieldset.halo_east:
            particle_dlon -= fieldset.halo_east - fieldset.halo_west  # noqa
        if particle.lat < fieldset.halo_south:
            particle_dlat += fieldset.halo_north - fieldset.halo_south  # noqa
        elif particle.lat > fieldset.halo_north:
            particle_dlat -= fieldset.halo_north - fieldset.halo_south  # noqa

    def slowlySouthWestward(particle, fieldset, time):
        particle_dlon -= 5 * particle.dt / 1e5  # noqa
        particle_dlat -= 3 * particle.dt / 1e5  # noqa

    kernels = pset.Kernel(AdvectionRK4)+slowlySouthWestward+periodicBC
    pset.execute(kernels, runtime=delta(days=6), dt=delta(hours=1))


def main(args=None):
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=2,
                   help='Number of particles to advect')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-f', '--fieldset', type=int, nargs=2, default=None,
                   help='Generate fieldset file with given dimensions')
    p.add_argument('-m', '--method', choices=('RK4', 'EE', 'RK45'), default='RK4',
                   help='Numerical method used for advection')
    args = p.parse_args(args)
    data_folder = download_example_dataset("MovingEddies_data")
    filename = str(data_folder / "moving_eddies")

    # Generate fieldset files according to given dimensions
    if args.fieldset is not None:
        fieldset = moving_eddies_fieldset(args.fieldset[0], args.fieldset[1], mesh='flat')
    else:
        fieldset = moving_eddies_fieldset(mesh='flat')
    outfile = "EddyParticle"

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("moving_eddies_example(fieldset, outfile, args.particles, mode=args.mode, \
                              verbose=args.verbose, method=method[args.method])",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        moving_eddies_example(fieldset, outfile, args.particles, mode=args.mode,
                              verbose=args.verbose, method=method[args.method])


if __name__ == "__main__":
    main()
