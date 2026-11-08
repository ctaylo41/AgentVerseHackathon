"""Smooth bounded actuator trajectories; measured-state checks at physics rate."""
import math

RATES = {'height': 0.35, 'truck': 0.45, 'dolly': 0.45, 'pan': 0.6, 'tilt': 0.5, 'roll': 0.35}


class MotionError(RuntimeError):
    pass


def check_limits(sim):
    for name, value in sim.configuration.items():
        lo, hi = sim.limits[name]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise MotionError(f'Measured {name}={value} is outside [{lo}, {hi}]')


def advance(sim, seconds, trace=None):
    """Check every physics step, not just rendered frames."""
    steps = max(1, math.ceil(seconds / sim.model.opt.timestep))
    for _ in range(steps):
        sim.checkpoint()
        sim.step(sim.model.opt.timestep)
        sim.path_validator.check(sim.data)
        check_limits(sim)
    if trace is not None:
        trace.append(dict(time=float(sim.data.time), joints=sim.configuration,
                          controls={name: float(sim.data.ctrl[sim.model.actuator(name).id]) for name in sim.axes}))


def move_to(sim, target, *, timeout=5.0, trace=None, callback=None):
    core = {'height', 'pan', 'tilt', 'roll'}
    if not (core <= set(target) <= set(sim.axes)):
        raise ValueError('Target must specify all core rig axes and no unknown axes')
    start = {name: float(sim.data.ctrl[sim.model.actuator(name).id]) for name in sim.axes}
    target = {name: float(target[name]) if name in target else start[name] for name in sim.axes}
    for name, value in target.items():
        lo, hi = sim.limits[name]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'Invalid {name} target: {value}')
    check_limits(sim)
    path = sim.path_validator.validate(sim.configuration, target)
    sim.path_validator.validate(start, target)
    # Quintic smoothstep maximum derivative is 1.875; rates bound setpoint speed.
    duration = max(0.1, max(1.875*abs(target[n]-start[n])/RATES[n] for n in sim.axes))
    steps = math.ceil(duration/0.02)
    for i in range(1, steps+1):
        t = i/steps
        blend = 10*t**3 - 15*t**4 + 6*t**5
        sim.command(**{n: start[n] + blend*(target[n]-start[n]) for n in sim.axes})
        advance(sim, duration/steps, trace)
        if callback and i % 5 == 0:
            callback(sim)
    stable, elapsed = 0.0, 0.0
    while elapsed < timeout:
        advance(sim, 0.02, trace)
        elapsed += 0.02
        close = all(abs(sim.configuration[n]-target[n]) <= (0.04 if n == 'height' else 0.015) for n in sim.axes)
        slow = all(abs(float(sim.data.joint(n).qvel[0])) < 0.015 for n in sim.axes)
        stable = stable + 0.02 if close and slow else 0
        if stable >= 0.2:
            return dict(settled=True, target=target, measured=sim.configuration, path_validation=path)
    raise MotionError('Actuators did not settle within the allowed time')


def step_sweep(sim, start, end, elapsed_before, dt, duration, trace=None):
    """Advance continuous in-shot camera motion along a rate-limited quintic curve."""
    if dt <= 0:
        return
    substeps = max(1, math.ceil(dt / 0.02))
    sub_dt = dt / substeps
    for i in range(1, substeps + 1):
        t = min(1.0, max(0.0, (elapsed_before + i * sub_dt) / max(duration, 1e-6)))
        blend = 10 * t**3 - 15 * t**4 + 6 * t**5
        sim.command(**{n: start[n] + blend * (end[n] - start[n]) for n in sim.axes})
        advance(sim, sub_dt, trace)

