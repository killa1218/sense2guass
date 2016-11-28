# --*-- coding:utf-8 --*--
import re

class BaseDataLoader:
    """ docstring for BaseDataLoader
        This class is used to define the common methods of data loaders.
    """
    def __init__(self):
        pass


    def load(self, datafiles):
        print('[ERROR] This method is not overrided.')
        pass


class LEDSDataLoader(BaseDataLoader):
    """ docstring for LEDSDataLoader
        This class is used to load the data of LEDS dataset, the datase contains pairs with pos tag.
    """
    PARSE_REGEX = re.compile('^(\w+)-(n|v|a|r)\t(\w+)-(n|v|a|r)\n$')

    def __init__(self):
        # self.load(datafiles)
        pass


    def load(self, datafiles):
        fp = open(datafiles, 'r')
        data = {}

        while True:
            line = fp.readline()

            if line == '':
                break
            else:
                match = self.PARSE_REGEX.match(line)

                if match:
                    hypo = match.group(1)
                    hypo_pos = match.group(2)
                    hyper = match.group(3)
                    hyper_pos = match.group(4)

                    if (hyper, hyper_pos) in data:
                        data[(hyper, hyper_pos)].append((hypo, hypo_pos))
                    else:
                        data[(hyper, hyper_pos)] = [(hypo, hypo_pos)]

        return data





