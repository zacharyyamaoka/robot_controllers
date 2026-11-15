// A C ABI over the header-only joint trajectory controller, for ctypes (controllers/joint_trajectory_cpp.py).
//
// WHY a C ABI and ctypes: no new Python dependency (pybind11/nanobind would be one) and no build backend
// change; the adapter compiles this one file on first use. Every array is caller-owned double[]; the
// trajectory arrays must stay alive while loaded (the adapter keeps references).
#include <ctime>
#include <new>

#include "robot_controllers/joint_trajectory_controller.hpp"

namespace {

struct Handle {
  robot_controllers::JointTrajectoryController controller;
  long long law_cpu_ns = 0;  // thread CPU time of the last tick's law, measured here (an upper bound: it
                             // includes half of each clock_gettime call)
};

long long thread_cpu_ns() {
  timespec now{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
  return static_cast<long long>(now.tv_sec) * 1000000000LL + now.tv_nsec;
}

}  // namespace

extern "C" {

// Bumped whenever a signature below changes; the adapter refuses a library of another version.
int rc_jtc_abi_version(void) { return 1; }

int rc_jtc_max_joints(void) { return static_cast<int>(robot_controllers::JointTrajectoryController::kMaxJoints); }

void* rc_jtc_create(void) { return new (std::nothrow) Handle(); }

void rc_jtc_destroy(void* handle) { delete static_cast<Handle*>(handle); }

// replace = 0 builds (active input stays), 1 is replace_input (clock restarts, active input + 1).
// Returns 0, or a robot_controllers::LoadError code (nothing changed).
int rc_jtc_load(void* handle, int replace, size_t samples, size_t joints, const double* times,
                const double* positions, const double* velocities, const double* effort,
                const double* path_tolerance_rad, const double* max_effort_nm) {
  auto& controller = static_cast<Handle*>(handle)->controller;
  const robot_controllers::TrajectoryView view{times, positions, velocities, effort, samples, joints};
  const auto error = replace ? controller.replace_input(view, path_tolerance_rad, max_effort_nm)
                             : controller.load(view, path_tolerance_rad, max_effort_nm);
  return static_cast<int>(error);
}

// One tick. Writes position, velocity and feed-forward (one per joint) and returns the status
// (0 running, 1 done, 2 aborted), or -1 when dt is invalid (nothing changed).
int rc_jtc_tick(void* handle, const double* measured_positions, double dt, double* out_position,
                double* out_velocity, double* out_effort) {
  auto* state = static_cast<Handle*>(handle);
  auto& controller = state->controller;
  const long long started = thread_cpu_ns();
  const bool ticked = controller.tick(measured_positions, dt);
  state->law_cpu_ns = thread_cpu_ns() - started;
  if (!ticked) return -1;
  const std::size_t joints = controller.joints();
  const double* position = controller.position();
  const double* velocity = controller.velocity();
  const double* effort = controller.feedforward_effort();
  for (std::size_t joint = 0; joint < joints; ++joint) {
    out_position[joint] = position[joint];
    out_velocity[joint] = velocity[joint];
    out_effort[joint] = effort[joint];
  }
  return static_cast<int>(controller.status());
}

int rc_jtc_status(const void* handle) {
  return static_cast<int>(static_cast<const Handle*>(handle)->controller.status());
}

double rc_jtc_clock(const void* handle) { return static_cast<const Handle*>(handle)->controller.clock_s(); }

double rc_jtc_end_time(const void* handle) { return static_cast<const Handle*>(handle)->controller.end_time(); }

int rc_jtc_active_input(const void* handle) {
  return static_cast<const Handle*>(handle)->controller.active_input();
}

int rc_jtc_worst_joint(const void* handle) { return static_cast<const Handle*>(handle)->controller.worst_joint(); }

double rc_jtc_worst_error(const void* handle) {
  return static_cast<const Handle*>(handle)->controller.worst_error();
}

long long rc_jtc_law_cpu_ns(const void* handle) { return static_cast<const Handle*>(handle)->law_cpu_ns; }

}  // extern "C"
