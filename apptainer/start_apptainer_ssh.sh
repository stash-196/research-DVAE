#!/bin/bash
apptainer instance start --bind /path/to/your/project:/workspace your_container.sif myinstance
apptainer exec instance://myinstance /usr/sbin/sshd -D