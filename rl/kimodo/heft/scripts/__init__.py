"""Staged verification for the HEFT-in-robocasa port.

Each script is independently falsifiable and they are meant to be run in order.
h0/h1 need nothing but this repo; h2-h5 need an mjlab trace and policy.pt from
the cluster; Kimodo is not connected until h6, after h5 has passed on a
HEFT-native clip. Connecting it earlier makes a failure unattributable among
bridge / tracker-usage / generation.
"""
