cd mesa
meson setup --reconfigure -Dteflon=true -Dvulkan-drivers= --prefix=/home/tomeu/src/mesa/install -Dgallium-drivers=etnaviv build
ninja -C build