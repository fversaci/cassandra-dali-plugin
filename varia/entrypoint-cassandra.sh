#! /bin/sh
service ssh start
exec /usr/local/bin/docker-entrypoint.sh "$@"
