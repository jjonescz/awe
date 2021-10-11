import parsel

def clean(page: parsel.Selector):
    page.css('script, style').remove()
    return page
