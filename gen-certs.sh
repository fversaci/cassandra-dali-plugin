#!/bin/sh
set -e # exit if any error occurs

mkdir -p certs && cd certs
# root CA
openssl req -new -x509 -noenc -keyout rootca.key -out rootca.crt -days 3650 \
	-subj "/C=IT/ST=Sardinia/L=Cagliari/O=CRS4/CN=rootCA"
openssl x509 -in rootca.crt -text -noout
# keystore
keytool -genkeypair -noprompt -keyalg RSA -keysize 2048 -validity 3650 -alias server -keystore keystore.p12 \
	-storepass keystore -dname "CN=server, O=CRS4, L=Cagliari, ST=Sardinia, C=IT"
keytool -list -keystore keystore.p12 -storepass keystore
keytool -keystore keystore.p12 -alias server -certreq -file server.csr -storepass keystore
openssl x509 -req -CA rootca.crt -CAkey rootca.key -in server.csr -out server.crt -days 3650 -CAcreateserial
openssl verify -CAfile rootca.crt server.crt
keytool -keystore keystore.p12 -alias rootca -importcert -file rootca.crt -noprompt -storepass keystore
keytool -keystore keystore.p12 -alias server -importcert -file server.crt -noprompt -storepass keystore
keytool -list -keystore keystore.p12 -storepass keystore
# truststore
keytool -keystore truststore.p12 -importcert -file rootca.crt -alias rootca -noprompt -storepass truststore
# client
openssl req -new -newkey RSA -nodes -out client.csr -keyout client.key -subj "/C=IT/ST=Sardinia/L=Cagliari/O=CRS4/CN=client"
openssl x509 -req -CA rootca.crt -CAkey rootca.key -in client.csr -out client.crt -days 3650 -CAcreateserial
openssl verify -CAfile rootca.crt client.crt
