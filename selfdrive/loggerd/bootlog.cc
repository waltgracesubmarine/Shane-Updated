#include <cassert>
#include <string>
#include <iostream>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/loggerd/logger.h"


static kj::Array<capnp::word> build_boot_log() {
  std::vector<std::string> bootlog_commands;
  if (Hardware::TICI()) {
    bootlog_commands.push_back("journalctl");
    bootlog_commands.push_back("sudo nvme smart-log --output-format=json /dev/nvme0");
  } else if (Hardware::EON()) {
    bootlog_commands.push_back("logcat -d");
  }

  MessageBuilder msg;
  auto boot = msg.initEvent().initBoot();

  boot.setWallTimeNanos(nanos_since_epoch());

  std::string pstore = "/sys/fs/pstore";
  std::map<std::string, std::string> pstore_map = util::read_files_in_dir(pstore);

  const std::vector<std::string> log_keywords = {"Kernel panic"};
  auto lpstore = boot.initPstore().initEntries(pstore_map.size());
  int i = 0;
  for (auto& kv : pstore_map) {
    auto lentry = lpstore[i];
    lentry.setKey(kv.first);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)kv.second.data(), kv.second.size()));
    i++;

    for (auto &k : log_keywords) {
      if (kv.second.find(k) != std::string::npos) {
        LOGE("%s: found '%s'", kv.first.c_str(), k.c_str());
      }
    }
  }

  // Gather output of commands
  i = 0;
  auto commands = boot.initCommands().initEntries(bootlog_commands.size());
  for (auto &command : bootlog_commands) {
    auto lentry = commands[i];

    lentry.setKey(command);

    const std::string result = util::check_output(command);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)result.data(), result.size()));

    i++;
  }

  boot.setLaunchLog(util::read_file("/tmp/launch_log"));
  return capnp::messageToFlatArray(msg);
}

int main(int argc, char** argv) {
  double t = millis_since_boot();
  clear_locks(LOG_ROOT);
  std::cout << (millis_since_boot() - t) / 1000. << std::endl;

  const std::string path = LOG_ROOT + "/boot/" + logger_get_route_name() + ".bz2";
  LOGW("bootlog to %s", path.c_str());
  std::cout << (millis_since_boot() - t) / 1000. << std::endl;

  // Open bootlog
  bool r = util::create_directories(LOG_ROOT + "/boot/", 0775);
  assert(r);
  std::cout << (millis_since_boot() - t) / 1000. << std::endl;

  BZFile bz_file(path.c_str());
  std::cout << (millis_since_boot() - t) / 1000. << std::endl;

  // Write initdata
  bz_file.write(logger_build_init_data().asBytes());
  std::cout << (millis_since_boot() - t) / 1000. << std::endl;

  // Write bootlog
  bz_file.write(build_boot_log().asBytes());
  std::cout << (millis_since_boot() - t) / 1000. << std::endl;

  return 0;
}
