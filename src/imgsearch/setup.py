import grp
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from imgsearch import config as cfg

SYSTEMD_SERVICE_TEMPLATE = """[Unit]
Description=iSearch - Lightweight Image Search Engine
After=network.target
Wants=network.target

[Service]
User={username}
Group={usergroup}
WorkingDirectory="{base_dir}"
Environment=PATH={py_bin}:/usr/local/bin:/usr/bin:/bin
ExecStart={py_bin}/isearch service start -b {base_dir} -m {model_key} -B {bind} -L {log_level}
Restart=on-failure
TimeoutStopSec=10s
SyslogIdentifier=isearch
OOMScoreAdjust=-500
MemoryMax=2560M
ProtectSystem=strict
ReadWritePaths="{base_dir}"
TemporaryFileSystem=/tmp:noexec

[Install]
WantedBy=multi-user.target
"""


LAUNCHD_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>{service_name}</string>

    <key>ProgramArguments</key>
    <array>
      <string>{py_bin}/isearch</string>
      <string>service</string>
      <string>start</string>
      <string>-b</string>
      <string>{base_dir}</string>
      <string>-m</string>
      <string>{model_key}</string>
      <string>-B</string>
      <string>{bind}</string>
      <string>-L</string>
      <string>{log_level}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{base_dir}</string>

    <key>EnvironmentVariables</key>
    <dict>
      <key>PATH</key>
      <string>{py_bin}:/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
      <key>SuccessfulExit</key>
      <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>{base_dir}/isearch.log</string>

    <key>StandardErrorPath</key>
    <string>{base_dir}/isearch.log</string>

    <key>ProcessType</key>
    <string>Background</string>

    <key>Nice</key>
    <integer>1</integer>
  </dict>
</plist>
"""


def get_env_variables(base_dir: str, model_key: str, bind: str, log_level: str, service_name: str):
    """Generate service config content for the current platform."""
    return {
        'base_dir': base_dir,
        'model_key': model_key,
        'bind': bind,
        'log_level': log_level,
        'service_name': service_name,
        'py_bin': str(Path(sys.executable).parent.absolute()),
        'username': os.getlogin(),
        'usergroup': grp.getgrgid(os.getgid()).gr_name,
    }


def setup_systemd_service(env_vars: dict[str, str]):
    """Setup systemd service for Linux."""
    # Create service config file
    config = SYSTEMD_SERVICE_TEMPLATE.format(**env_vars)
    with NamedTemporaryFile(mode='w', suffix='.service', delete=False) as tmp:
        tmp.write(config)
        tmp_path = tmp.name
    subprocess.run(['sudo', 'mv', tmp_path, f'/etc/systemd/system/{env_vars["service_name"]}'], check=True)

    # Enable and start service
    subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
    subprocess.run(['sudo', 'systemctl', 'enable', env_vars['service_name']], check=True)
    subprocess.run(['sudo', 'systemctl', 'start', env_vars['service_name']], check=True)


def setup_launchd_service(env_vars: dict[str, str]):
    """Setup launchd service for macOS."""
    # Create launchd service file
    config = LAUNCHD_PLIST_TEMPLATE.format(**env_vars)
    service_file = Path.home() / 'Library/LaunchAgents' / f'{env_vars["service_name"]}.plist'
    if service_file.exists():
        raise FileExistsError(f'Service file {service_file} already exists.')
    service_file.parent.mkdir(parents=True, exist_ok=True)
    service_file.write_text(config)

    # Load and start service
    subprocess.run(['launchctl', 'load', service_file], check=True)
    subprocess.run(['launchctl', 'start', env_vars['service_name']], check=True)


def setup_service(
    base_dir: str | Path = cfg.BASE_DIR,
    model_key: str = cfg.DEFAULT_MODEL_KEY,
    bind: str = cfg.UNIX_SOCKET,
    log_level: str = 'info',
    service_name=cfg.SERVICE_NAME,
) -> bool:
    """Install service on the current platform."""
    base_dir = Path(base_dir).resolve()
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
    if not base_dir.is_dir():
        raise NotADirectoryError(f'Base directory {base_dir} does not exist.')

    env_vars = get_env_variables(str(base_dir), model_key, bind, log_level, service_name)

    if sys.platform == 'linux':
        setup_systemd_service(env_vars)
        return True
    elif sys.platform == 'darwin':
        setup_launchd_service(env_vars)
        return True
    else:
        raise OSError(f'Service management not supported on platform: {sys.platform}')


def remove_systemd_service(service_name: str):
    """Remove systemd service for Linux."""
    # Stop and disable service
    subprocess.run(['sudo', 'systemctl', 'stop', service_name], check=False)
    subprocess.run(['sudo', 'systemctl', 'disable', service_name], check=False)

    # Remove service file
    service_file = Path('/etc/systemd/system/' + service_name)
    if service_file.exists():
        subprocess.run(['sudo', 'rm', str(service_file)], check=True)

    # Reload daemon
    subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)


def remove_launchd_service(service_name: str):
    """Remove launchd service for macOS."""
    # Stop service
    subprocess.run(['launchctl', 'stop', service_name], check=False)

    # Unload and remove plist
    plist_path = Path.home() / 'Library/LaunchAgents' / f'{service_name}.plist'
    if plist_path.exists():
        subprocess.run(['launchctl', 'unload', str(plist_path)], check=False)
        plist_path.unlink()


def remove_service(service_name=cfg.SERVICE_NAME) -> bool:
    """Remove service from the current platform."""

    if sys.platform == 'linux':
        remove_systemd_service(service_name)
        return True
    elif sys.platform == 'darwin':
        remove_launchd_service(service_name)
        return True
    else:
        raise OSError(f'Service management not supported on platform: {sys.platform}')
