"""Small MJCF XML geometry helpers."""

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.mjcf_utils import string_to_array


def quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def quat_to_mat(quat):
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotated_bounds(center, half_size, quat):
    center = np.asarray(center, dtype=float)
    half_size = np.asarray(half_size, dtype=float)
    mat = quat_to_mat(quat)
    points = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                points.append(mat @ (center + half_size * np.array([sx, sy, sz])))
    points = np.asarray(points, dtype=float)
    bounds_center = (points.min(axis=0) + points.max(axis=0)) / 2
    bounds_half = (points.max(axis=0) - points.min(axis=0)) / 2
    return bounds_center, bounds_half, float(points[:, 2].min())


def mjcf_xml_bounds(xml_path):
    root = ET.parse(xml_path).getroot()
    body = root.find("./worldbody/body")
    points = []

    def parse_vec(value, default):
        return string_to_array(value) if value is not None else np.asarray(default)

    def collect(item, parent_pos, parent_quat):
        pos = parent_pos + quat_to_mat(parent_quat) @ parse_vec(
            item.get("pos"), [0.0, 0.0, 0.0]
        )
        quat = quat_multiply(
            parent_quat, parse_vec(item.get("quat"), [1.0, 0.0, 0.0, 0.0])
        )
        mat = quat_to_mat(quat)
        points.append(pos)
        for geom in item.findall("./geom"):
            if geom.get("type") != "box":
                continue
            center = pos + mat @ parse_vec(geom.get("pos"), [0.0, 0.0, 0.0])
            geom_quat = quat_multiply(
                quat, parse_vec(geom.get("quat"), [1.0, 0.0, 0.0, 0.0])
            )
            geom_mat = quat_to_mat(geom_quat)
            size = parse_vec(geom.get("size"), [0.02, 0.02, 0.02])
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        points.append(
                            center + geom_mat @ (size * np.array([sx, sy, sz]))
                        )
        for child in item.findall("./body"):
            collect(child, pos, quat)

    if body is None:
        return np.zeros(3), np.full(3, 0.05), -0.05
    collect(body, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    points = np.asarray(points, dtype=float)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    half_size = np.maximum((points.max(axis=0) - points.min(axis=0)) / 2, 0.02)
    return center, half_size, float(points[:, 2].min())
