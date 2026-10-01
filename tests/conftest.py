import pytest


@pytest.fixture
def mounts_file(tmp_path):
    """Fake /proc/mounts: tmp_path mounted as a Windows drive (9p/drvfs)."""

    def make(fs_type: str = "9p", mount_point: str | None = None):
        point = mount_point or str(tmp_path)
        f = tmp_path / "mounts"
        f.write_text(
            "rootfs / ext4 rw 0 0\n"
            f"G:\\134 {point} {fs_type} rw,noatime,aname=drvfs;path=G:\\ 0 0\n"
        )
        return f

    return make
