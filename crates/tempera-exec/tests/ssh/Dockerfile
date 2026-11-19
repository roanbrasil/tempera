# Throwaway sshd for the SSH executor integration test (tests/ssh_docker.rs).
# The test injects a freshly generated public key through AUTHORIZED_KEY.
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server rsync ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && ssh-keygen -A \
    && useradd --create-home --shell /bin/bash tempera \
    && usermod -p '*' tempera \
    && printf '%s\n' 'PasswordAuthentication no' 'KbdInteractiveAuthentication no' \
        'PermitRootLogin no' 'AllowTcpForwarding yes' > /etc/ssh/sshd_config.d/tempera.conf

COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
