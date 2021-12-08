from awe.data import swde

sds = swde.Dataset(suffix='-exact')
sds.validate(
    parallelize=None,
    skip=0,
    verticals=sds.verticals[:1],
    end_after_first_error=True,
    collect_errors=True,
    error_callback=lambda i, _, e: print(f'{i}: {str(e)}'),
    save_list=True,
    read_list=True,
)
