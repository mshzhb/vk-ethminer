#!/usr/bin/env python3
import os
import platform

version = '0.1'
if os.name == 'nt':
    os.system('rmdir /s build-release')
    os.system('mkdir build-release')
    os.chdir('build-release/')
    os.system('cmake ..')
    os.system('cmake --build . --config Release')
    os.system('Xcopy shaders Release\shaders\ /E /H /C /I')
    os.chdir('Release/')
    with open("etc_2miners_cn.bat", "w") as file:
        file.write("::replace 0x... with your wallet\n")
        file.write("vulkan_ethminer.exe --server asia-etc.2miners.com --port 1010 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle\n")
    with open("etc_ethermine_asia.bat", "w") as file:
        file.write("::replace 0x... with your wallet\n")
        file.write("vulkan_ethminer.exe --server asia1-etc.ethermine.org --port 4444 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle\n")
    with open("etc_ethermine_eu.bat", "w") as file:
        file.write("::replace 0x... with your wallet\n")
        file.write("vulkan_ethminer.exe --server eu1-etc.ethermine.org --port 4444 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle\n")
    with open("etc_ethermine_us.bat", "w") as file:
        file.write("::replace 0x... with your wallet\n")
        file.write("vulkan_ethminer.exe --server us1-etc.ethermine.org --port 4444 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle\n")
    os.system('tar.exe -a -c -f vulkan-ethminer-v{}-windows-amd64.zip vulkan_ethminer.exe shaders *.bat'.format(version))
else:
    os.system('rm -rf build-release')
    os.system('mkdir build-release')
    os.chdir('build-release/')
    os.system('pwd')
    os.system('cmake -DCMAKE_BUILD_TYPE=Release ..')
    os.system('make')
    with open("etc_2miners_cn.sh", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("#replace 0x... with your wallet\n")
        file.write("./vulkan_ethminer --server asia-etc.2miners.com --port 1010 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle\n")
    with open("etc_ethermine_asia.sh", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("#replace 0x... with your wallet\n")
        file.write("./vulkan_ethminer --server asia1-etc.ethermine.org --port 4444 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle\n")
    with open("etc_ethermine_eu.sh", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("#replace 0x... with your wallet\n")
        file.write("./vulkan_ethminer --server eu1-etc.ethermine.org --port 4444 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle\n")
    with open("etc_ethermine_us.sh", "w") as file:
        file.write("#!/bin/bash\n")
        file.write("#replace 0x... with your wallet\n")
        file.write("./vulkan_ethminer --server us1-etc.ethermine.org --port 4444 --wallet 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A --rig miner --shader wave_shuffle\n")
    os.system('chmod +x *.sh')
    if platform.system() == 'Linux':
        os.system('tar cvzf vulkan-ethminer-v{0}-linux-amd64.tar.gz vulkan_ethminer shaders *.sh'.format(version))
    else:
        os.system('tar cvzf vulkan-ethminer-v{0}-macOS-arm64.tar.gz vulkan_ethminer shaders *.sh'.format(version)) 
        
