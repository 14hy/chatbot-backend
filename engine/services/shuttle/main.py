import enum
from datetime import time, datetime, timedelta


class Table(enum.Enum):
    # 0: 학기 중, 1: 계절학기, 2: 방학 중
    # 0: 월~금, 1: 휴일
    # 0: 순환, 1: 한대앞역, 2: 예술인
    학기중_월금_순환노선 = [[5, 3, 1], [6, 5, 2]]
    학기중_월금_예술인 = [[0, 2, 0]]
    학기중_월금_한대앞 = [[2, 0, 3], [4, 1, 3]]  # 0,0,2
    학기중_휴일_순환노선 = [[3, 4, 3]]  # 0,1,0
    계절_월금_순환노선 = [[5, 4, 2]]  # 1,0,0
    계절_월금_예술인 = [[1, 2, 0]]  # 1,0,1
    계절_월금_한대앞 = [[1, 0, 3], [4, 1, 3]]  # 1,0,2
    계절_휴일_순환노선 = [[3, 4, 3]]  # 1,0,0
    방학_월금_순환노선 = [[1, 4, 3]]  # 2,0,0
    방학_휴일_순환노선 = [[3, 4, 4]]  # 2,1,0


class TimeTable(object):
    """
    # 정류장의 종류
    1. 창의인재원
    2. 셔틀콕
    3. 한대앞역
    4. 예술인APT
    # 노선
    ## 순환
    창의인재원 -> 셔틀콕 -> 한대앞역 -> 예술인APT -> 셔틀콕 -> 창의인재원
    ## 예술인 APT
    창의인재원 -> 셔틀콕 -> 예술인APT -> 셔틀콕 -> 창의인재원
    ## 한대앞역
    창의인재원 -> 셔틀콕 -> 한대앞역 -> 셔틀콕 -> 창의인재원
    # 소요시간
    창의인재원 <-> 셔틀콕 (5분)
    셔틀콕 <-> 한대앞역, 예술인APT (10분)
    한대앞역 <-> 예술인APT (5분)
    """

    # 학기 중, 계절학기 첫 차는 순환 운행
    # 시작시간, 끝나는 시간, 시즌, 휴일, 노선에 따라 차등 적용

    def __init__(self):
        pass
        self.start_time = [self.create_delta(50, 7), self.create_delta(0, 8),  # 0, 1
                           self.create_delta(20, 8), self.create_delta(0, 9),  # 2, 3
                           self.create_delta(0, 13), self.create_delta(0, 19),  # 4, 5
                           self.create_delta(45, 21)]  # 6
        self.end_time = [self.create_delta(20, 12), self.create_delta(30, 18),  # 0, 1
                         self.create_delta(50, 18), self.create_delta(30, 21),  # 2, 3
                         self.create_delta(0, 22), self.create_delta(0, 23)]  # 4, 5
        self.interval = [self.create_delta(5, 0), self.create_delta(10, 0),  # 0, 1
                         self.create_delta(15, 0), self.create_delta(30, 0),  # 2, 3
                         self.create_delta(0, 1)]  # 4

        self.recipe = [[[Table.학기중_월금_순환노선.value, Table.학기중_월금_예술인.value, Table.학기중_월금_한대앞.value],
                        [Table.학기중_휴일_순환노선.value]],
                       [[Table.계절_월금_순환노선.value, Table.계절_월금_예술인.value, Table.계절_월금_한대앞.value],
                        [Table.계절_휴일_순환노선.value]], [[Table.방학_월금_순환노선.value], [Table.방학_휴일_순환노선.value]]]

    def get_table(self, season, weekend):
        '''

        :return: [순환노선, 에술인, 한대앞역], 셔틀콕 기준 시간
        순환노선: 창의인재원(-5분), 셔틀콕, 한대앞역(10분), 예술인(15분), 셔틀콕2(25분), 창의인재원2(30분) # 35분코스
        예술인: 창의인재원(-5분), 셔틀콕, 예술인(10분), 셔틀콕2(20분), 창의인재원2(25분) # 30분 코스
        한대앞역: 창의인재원(-5분), 셔틀콕, 한대앞역(10분), 셔틀콕2(20분), 창의인재원2(25분) # 30분 코스
        '''

        output = []
        recipe = self.recipe[season][weekend]

        for i in range(len(recipe)):
            temp = []
            for j in range(len(recipe[i])):
                temp2 = []
                s = self.start_time[recipe[i][j][0]]
                e = self.end_time[recipe[i][j][1]]
                t = self.interval[recipe[i][j][2]]
                while s <= e:
                    temp2.append(s)
                    s += t
                temp.append(temp2)
            output.append(temp)

        return output

    def response(self):
        '''
        가장 가까운 셔틀 시간을 계산하여 뿌려 줌
        :return:
        '''
        current_time = timedelta(hours=datetime.today().hour,
                                 minutes=datetime.today().minute,
                                 seconds=datetime.today().second)
        print('현재 시각은 {} 입니다.'.format(current_time))
        season = self.season_is(current_time)
        weekend = self.weekend_is(current_time)
        # table = self.get_table(season, weekend)
        table = self.get_table(0, 0)
        output = self.calc_response(table, current_time)

    def season_is(self, current_time):
        '''
        학기중/ 계절학기/ 방학 인지
        :param time:
        :return:
        '''
        pass

    def weekend_is(self, current_time):
        '''
        월~금/ 주말, 공휴일 인지
        :return:
        '''
        return 0

    @staticmethod
    def create_time(hours, minutes=0):
        return time(hour=hours, minute=minutes)

    @staticmethod
    def create_delta(minutes, hours=0):
        return timedelta(minutes=minutes, hours=hours)

    def calc_response(self, table, current_time):
        '''
        [창의원, 셔틀콕, 한대앞역, 예술인, 셔틀콕2]
        :param table:
        :param current_time:
        :return:
        '''

        # 창의원
        gap = - self.create_delta(5)
        dorm_cycle, dorm_station, dorm_artin = self.calc_close_time(table, current_time, gap)
        # None 이 들어올 시, 예외처리
        # 셔틀콕
        gap = self.create_delta(0)
        shuttle1_cycle, shuttle1_station, shuttle1_artin = self.calc_close_time(table, current_time, gap)
        # 한대앞역
        gap = self.create_delta(10)
        station_cycle, station_station, _ = self.calc_close_time(table, current_time, gap)
        # 예술인
        gap = self.create_delta(10)
        _, _, artin_artin = self.calc_close_time(table, current_time, gap)
        gap = self.create_delta(15)
        artin_cycle, _, _ = self.calc_close_time(table, current_time, gap)
        # 셔틀콕2
        gap = self.create_delta(20)
        _, shuttle2_station, shuttle2_artin = self.calc_close_time(table, current_time, gap)
        gap = self.create_delta(25)
        shuttle2_cycle, _, _= self.calc_close_time(table, current_time, gap)

        self.print_time('기숙사에서 출발하는 순환버스', dorm_cycle)
        self.print_time('기숙사에서 출발하는 한대앞역행 버스', dorm_station)
        self.print_time('기숙사에서 출발하는 예술인행 버스', dorm_artin)
        self.print_time('셔틀콕에서 출발하는 순환버스', shuttle1_cycle)
        self.print_time('셔틀콕에서 출발하는 한대앞역행 버스', shuttle1_station)
        self.print_time('셔틀콕에서 출발하는 예술인행 버스', shuttle1_artin)
        self.print_time('한대앞역에서 출발하는 버스', self.close_bus(station_cycle, station_station))
        self.print_time('예술인아파트에서 출발하는 버스', self.close_bus(artin_cycle, artin_artin))
        self.print_time('셔틀콕에서 기숙사로 가는 버스', self.close_bus(shuttle2_station, shuttle2_cycle, shuttle2_artin))


    def close_bus(self, a, b, c=None):
        if a is None:
            a = 99999
        if b is None:
            b = 99999
        if c is None:
            c = 99999

        return min(a,b,c)

    def print_time(self, text, close_time):
        if close_time is None:
            print(text,'는 운행이 종료 되었습니다.')
            return
        m, s = self.pretty(close_time)
        print(text,'는 {}분 {}초 후에 도착합니다.'.format(m, s))


    def pretty(self, seconds):

        m = 0
        s = seconds
        while s >= 60:
            s -= 60
            m += 1

        return m, s

    def calc_close_time(self, table, current_time, gap):
        table_gap = self.add_gap(table, gap)
        cycle = self.get_close_time(table_gap[0], current_time)
        station = self.get_close_time(table_gap[1], current_time)
        artin = self.get_close_time(table_gap[2], current_time)
        return cycle, station, artin


    def add_gap(self, table, gap):
        '''

        :param table:
        :param gap:
        :return:
        '''

        output = table.copy()
        for i in range(len(table)):
            for j in range(len(table[i])):
                for k in range(len(table[i][j])):
                    output[i][j][k] = table[i][j][k] + gap

        return output

    def get_close_time(self, table, current_time):
        """

        :param table:
        :param current_time:
        :return:
        """
        for i in range(len(table)):
            for j in range(len(table[i])):
                days = (table[i][j] - current_time).days
                seconds = (table[i][j].seconds - current_time.seconds)
                if seconds > 0 and days != -1:
                    return seconds
        return None


if __name__ == "__main__":
    test = TimeTable()
    test.response()
