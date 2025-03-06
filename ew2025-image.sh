set -x -e

CHROOT=./ew2025-image
RUN="sudo systemd-nspawn -D $CHROOT"
RUN_USER="sudo systemd-nspawn -D $CHROOT -u user --chdir=/home/user"
EXTRA_PACKAGES=openssh-server,sudo,kmod,linux-base,netbase,dhcpcd-base,dbus,ifupdown,net-tools,python3,python3-pip,git,python3-libcamera,udev,libcamera-ipa,dbus-user-session,vim

MESA_SRC=./mesa/
MESA_REPO=https://gitlab.freedesktop.org/tomeu/mesa.git
SSH_ID_PUB=$HOME/.ssh/id_rsa.pub

if [ ! -d $CHROOT ]; then

   # Setup root filesystem
   mkdir $CHROOT
   sudo debootstrap --arch=arm64 --include=$EXTRA_PACKAGES testing $CHROOT http://deb.debian.org/debian/
   sudo mkdir $CHROOT/lib/modules

   # Setup chroot
   sudo cp /etc/hosts $CHROOT/etc/hosts
   sudo cp /proc/mounts $CHROOT/etc/mtab

   # Setup host name
   sudo su -c "echo ew2025 > $CHROOT/etc/hostname"
   sudo su -c "cat > $CHROOT/etc/hosts" << EOF
127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4 ew2025
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6 ew2025
EOF

   # Setup user
   $RUN adduser --comment "" user
   $RUN usermod -aG sudo user
   $RUN usermod -aG video user
   $RUN usermod -aG render user
   $RUN usermod -aG input user
   sudo cp -rf MappedFrameBuffer.py cam-kms-tf.py dmabufsync.py $CHROOT/home/user/.
   sudo cp -rf labelmap.txt ssdlite_mobiledet_coco_qat_postprocess.tflite build-mesa.sh run-demo.sh $CHROOT/home/user/.
   sudo cp -rf tflite_runtime-2.16.2-cp313-cp313-linux_aarch64.whl $CHROOT/home/user/.
   sudo su -c "echo user ALL=\(ALL\) NOPASSWD: ALL >> $CHROOT/etc/sudoers"

   # Setup systemd unit to run demo at boot
   sudo cp ew2025.service $CHROOT/etc/systemd/system/
   sudo ln -s $CHROOT/etc/systemd/system/ $CHROOT//etc/systemd/system/multi-user.target.wants/ew2025.service

   # Setup SSH
   $RUN_USER mkdir /home/user/.ssh
   $RUN_USER chmod 700 /home/user/.ssh
   sudo cat $SSH_ID_PUB > $CHROOT/home/user/.ssh/authorized_keys
   $RUN_USER chmod 600 /home/user/.ssh/authorized_keys

   # Setup demo
   $RUN apt-get update
   $RUN_USER python3 -m pip -v install --break-system-packages /home/user/tflite_runtime-2.16.2-cp313-cp313-linux_aarch64.whl
   $RUN_USER python3 -m pip -v install --break-system-packages opencv-python-headless
   $RUN_USER python3 -m pip -v install --break-system-packages git+https://github.com/tomba/pykms.git

   # Build Mesa3D
   sudo su -c "echo deb-src http://deb.debian.org/debian testing main >> $CHROOT/etc/apt/sources.list"
   $RUN apt-get update
   $RUN apt-get -y build-dep mesa

   # Clone Tomeu's mesa fork if a 'mesa' directory is not available
   if [ ! -d $MESA_SRC ] ; then
	   git clone $MESA_REPO -b imx8mp-demo-ew2025 $MESA_SRC
   fi
   sudo rsync -av $MESA_SRC $CHROOT/home/user/mesa
   $RUN chown -R user:user /home/user/mesa

   $RUN_USER bash ./build-mesa.sh
fi
