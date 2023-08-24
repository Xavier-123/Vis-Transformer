data = dict(
    path='',
    train='',
    val='',
    test='',
    names={0: "cat"},
    yaml_file='',
    nc=1,
)

args = dict(
    model='l',
    data=data,
    # device=device,
    device='0',
    imgsz=196,
    exist_ok=True,
    batch=1,
    deterministic=False,
    amp=False)
