import sys
import time


class ProgressHelper(object):
    def __init__(self, name, comment=''):
        self.name = name
        self.comment = comment
        self.start = time.time()

    def __enter__(self):
        print("""<filter-start>
            <filter-name>%s</filter-name>
            <filter-comment>%s</filter-comment>
            </filter-start>""" % (self.name, self.comment))
        sys.stdout.flush()
        self.start = time.time()
        return self

    def progress(self, val):
        print("""<filter-progress>%s</filter-progress>""" % val)
        sys.stdout.flush()

    def message(self, comment):
        self.comment = comment
        print("""<filter-comment>%s</filter-comment>""" % comment)
        sys.stdout.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        duration = end - self.start
        print("""<filter-end>
             <filter-name>%s</filter-name>
             <filter-time>%s</filter-time>
            </filter-end>""" % (self.name, duration))
        sys.stdout.flush()


__all__ = ['ProgressHelper']
