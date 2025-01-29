#! /bin/bash
set -e
ssh-keyscan -H cassandra >> ~/.ssh/known_hosts
exec /opt/nvidia/nvidia_entrypoint.sh "$@"
