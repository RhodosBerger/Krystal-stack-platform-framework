# Krystal-Stack Janet DSL // WSL Emulation Subsystem & Virtual Linux Runtime
# Riadi virtuálne linuxové prostredie, simulátor balíčkovača DNF a mostík medzi fyzickým WSL2 a emulátorom.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)

(def SUPPORTED-DISTROS
  @{:fedora-40
    @{:name "Fedora Linux 40 (Cloud Edition)" :pkg-mgr "dnf" :family "rhel" :vital-max-hp VITAL-MAX-HP}
    :rhel-9
    @{:name "Red Hat Enterprise Linux 9.4" :pkg-mgr "dnf" :family "rhel" :vital-max-hp VITAL-MAX-HP}
    :ubuntu-26
    @{:name "Ubuntu 26.04 LTS" :pkg-mgr "apt" :family "debian" :vital-max-hp VITAL-MAX-HP}
    :alpine-320
    @{:name "Alpine Linux v3.20" :pkg-mgr "apk" :family "busybox" :vital-max-hp VITAL-MAX-HP}})

(def CANONICAL-PACKAGES
  @{:dnf @{:size-kib 1024 :category "system" :installed true}
    :bash @{:size-kib 2048 :category "shell" :installed true}
    :python3 @{:size-kib 15360 :category "runtime" :installed true}
    :assimp @{:size-kib 8192 :category "3d-tools" :installed false}
    :blender @{:size-kib 245760 :category "3d-modeling" :installed false}
    :vulkan-tools @{:size-kib 2048 :category "gpu-tools" :installed false}})

(defn get-distro-profile
  "Returns the configuration profile for a given Linux distribution key."
  [key]
  (get SUPPORTED-DISTROS key))

(defn evaluate-wsl-command
  "Evaluates virtual POSIX command line and returns status code and mode."
  [cmd-str is-physical]
  @{:command cmd-str
    :mode (if is-physical "PHYSICAL_WSL2" "EMULATED_WSL_SUBSYSTEM")
    :vital-max-hp VITAL-MAX-HP
    :exit-code 0})

(defn resolve-package-dependencies
  "Simulates DNF transaction resolution."
  [pkg-key]
  (let [pkg (get CANONICAL-PACKAGES pkg-key)]
    (if pkg
      @{:status "RESOLVED" :pkg pkg-key :vital-max-hp VITAL-MAX-HP}
      @{:status "UNKNOWN_PACKAGE" :pkg pkg-key :vital-max-hp VITAL-MAX-HP})))
