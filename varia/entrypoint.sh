#!/bin/bash

# update user id
set -e
if [[ ! -f "/home/user/.usermod" ]]; then
    sudo usermod -u $EXT_UID user 
    sudo find /cassandra/ -user 1000 -exec chown -h user {} \; &&  sudo find /home/user/ -user 1000 -exec chown -h user {} \; 
    touch "/home/user/.usermod"
fi

# configure and start cassandra server
pgrep -f /cassandra/bin || ( /cassandra/bin/cassandra 2>&1 | grep "state jump to NORMAL" && until nc -z 127.0.0.1 9042; do sleep 1; done && sleep 10 && echo 'Port 9042 is now open' )
echo "Node is now up and running"
touch /cassandra/logs/system.log
tail -f /cassandra/logs/system.log
