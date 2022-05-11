import os

from cereal import car
from common.params import Params
from selfdrive.hardware import TICI, PC
from selfdrive.manager.process import PythonProcess, NativeProcess, DaemonProcess
from common.op_params import opParams

WEBCAM = os.getenv("USE_WEBCAM") is not None

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  return params.get_bool("IsDriverViewEnabled")

def notcar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return CP.notCar

def logging(started, params, CP: car.CarParams) -> bool:
  run = (not CP.notCar) or not params.get_bool("DisableLogging")
  return started and run

procs = [
  DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),
  # due to qualcomm kernel bugs SIGKILLing camerad sometimes causes page table corruption
  NativeProcess("camerad", "selfdrive/camerad", ["./camerad"], unkillable=True, callback=driverview, sentry_mode=True),
  NativeProcess("clocksd", "selfdrive/clocksd", ["./clocksd"]),
  NativeProcess("dmonitoringmodeld", "selfdrive/modeld", ["./dmonitoringmodeld"], enabled=(not PC or WEBCAM), callback=driverview),
  NativeProcess("logcatd", "selfdrive/logcatd", ["./logcatd"]),
  NativeProcess("loggerd", "selfdrive/loggerd", ["./loggerd"], onroad=False, callback=logging, sentry_mode=True),
  NativeProcess("modeld", "selfdrive/modeld", ["./modeld"]),
  NativeProcess("navd", "selfdrive/ui/navd", ["./navd"], offroad=True),
  NativeProcess("proclogd", "selfdrive/proclogd", ["./proclogd"]),
  NativeProcess("sensord", "selfdrive/sensord", ["./sensord"], enabled=not PC, offroad=True),
  NativeProcess("ubloxd", "selfdrive/locationd", ["./ubloxd"], enabled=(not PC or WEBCAM)),
  NativeProcess("ui", "selfdrive/ui", ["./ui"], offroad=True, watchdog_max_dt=(5 if TICI else None)),
  NativeProcess("soundd", "selfdrive/ui/soundd", ["./soundd"], offroad=True),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"]),
  NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd"),
  PythonProcess("controlsd", "selfdrive.controls.controlsd"),
  PythonProcess("deleter", "selfdrive.loggerd.deleter", offroad=True),
  PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", enabled=(not PC or WEBCAM), callback=driverview),
  PythonProcess("logmessaged", "selfdrive.logmessaged", offroad=True),
  PythonProcess("pandad", "selfdrive.boardd.pandad", offroad=True),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd"),
  PythonProcess("plannerd", "selfdrive.controls.plannerd"),
  PythonProcess("radard", "selfdrive.controls.radard"),
  PythonProcess("thermald", "selfdrive.thermald.thermald", offroad=True),
  PythonProcess("timezoned", "selfdrive.timezoned", enabled=TICI, offroad=True),
  PythonProcess("tombstoned", "selfdrive.tombstoned", enabled=not PC, offroad=True),
  PythonProcess("updated", "selfdrive.updated", enabled=not PC, onroad=False, offroad=True),
  PythonProcess("uploader", "selfdrive.loggerd.uploader", offroad=True),
  PythonProcess("statsd", "selfdrive.statsd", offroad=True),

  NativeProcess("bridge", "cereal/messaging", ["./bridge"], onroad=False, callback=notcar),
  PythonProcess("webjoystick", "tools.joystick.web", onroad=False, callback=notcar),

  # Experimental
  PythonProcess("rawgpsd", "selfdrive.sensord.rawgps.rawgpsd", enabled=os.path.isfile("/persist/comma/use-quectel-rawgps")),

  # Stock Additions daemons
  PythonProcess("lanespeedd", "selfdrive.controls.lane_speed"),
  PythonProcess("sentryd", "selfdrive.sentryd", onroad=False, offroad=True),
]

if opParams().get('update_behavior').lower().strip() != 'off' and not os.path.exists('/data/no_ota_updates'):
  procs.append(PythonProcess("updated", "selfdrive.updated", enabled=not PC, onroad=False, offroad=True))

managed_processes = {p.name: p for p in procs}
