from utils.Loader import dataPreLoader


def process(line):
    return line['Category'].to_list()


class DataStruct:

    def __init__(self, data):
        self.edgeLeft = data['Pos_b'].tolist()
  