import numpy as np
from datetime import datetime, timedelta
import math
from copy import deepcopy
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


InputVariables = pd.read_excel(r'E:\KLTN_DUU\Source\Data\Data for Coding.xlsx', sheet_name='Input Variables')
VS = pd.read_excel(r'E:\KLTN_DUU\Source\Data\Data for Coding.xlsx', sheet_name='Vessel Schedule')
ChannelBerth = pd.read_excel(r'E:\KLTN_DUU\Source\Data\Data for Coding.xlsx', sheet_name='Channel-Berth')
Tidal = pd.read_excel(r'E:\KLTN_DUU\Source\Data\Data for Coding.xlsx', sheet_name='Tidal')
VesselSchedule = VS[:20]
LengthSave = pd.read_excel(r'E:\KLTN_DUU\Source\Data\Data for Coding.xlsx', sheet_name='Khoảng cách an toàn')
channel_tide = pd.read_excel(r'E:\KLTN_DUU\Source\Data\Data for Coding.xlsx', sheet_name='Channel Tide')
berth_tide = pd.read_excel(r'E:\KLTN_DUU\Source\Data\Data for Coding.xlsx', sheet_name='Berth Tide')

LSd = LengthSave['Khoảng cách d giữa các tàu (m)'].tolist()
LSe = LengthSave['Khoảng cách e giữa tàu và điểm cuối đoạn thẳng tuyến bến (m)'].tolist()

TG0 = InputVariables.iloc[0, 2] #Thời gian cố định để tàu di chuyển qua kênh
W = InputVariables.iloc[1, 2] #Độ dài mỗi khung thời gian thủy triều cao/thấp
R = InputVariables.iloc[2, 2] #Một số nguyên đủ lớn để ràng buộc
M = InputVariables.iloc[3, 2] #Tổng số xe tải có thể sử dụng
CE0 = InputVariables.iloc[4, 2]  #thời gian để 1 cần cẩu bốc dỡ 1 container (giờ)
CF0 = InputVariables.iloc[5, 2]  #thời gian để 1 cần cẩu bốc dỡ 1 container (giờ)
vh = np.random.normal(loc=25, scale=4) #Tốc độ xe tải khi không tải
vl = np.random.normal(loc=18, scale=3) #Tốc độ khi xe container quá tải
n_ = InputVariables.iloc[9, 2] #Mức tiêu thụ nhiên liệu khi chạy không tải
E1 = float(InputVariables.iloc[10, 2]) #Hệ số phát thải carbon của xe tải
E2 = InputVariables.iloc[11, 2] #Hệ số chuyển đổi điện năng sang khí thải carbon
lambda1 = InputVariables.iloc[12, 2] #Mức tiêu thụ năng lượng khi cần cẩu làm việc
lambda2 = InputVariables.iloc[13, 2] #Mức tiêu thụ năng lượng khi cần cẩu di chuyển giữa các vị trí
Channel = InputVariables.iloc[14, 2] #Kênh
V_name = VesselSchedule['Vessel Name'].tolist() #Vessel
TOi = VesselSchedule['TOi (Estimated Time of Arrival)'].tolist() # Thời gian dự kiến tàu đến
TFi = VesselSchedule['TFi (Estimated Time of departure)'].tolist() # Thời gian dự kiến tàu đi
TWi = VesselSchedule['TWi (Maximum waiting time, hour)'].tolist() # Thời gian chờ tối đa của tàu
VLi = VesselSchedule['VLi (Length, m)'].tolist() #VLi (Chiều dài tàu i) (mét)
VDi = VesselSchedule['VDi (Draft, m)'].tolist() #VDi (Mớn nước của tàu i) (mét)
VEi = VesselSchedule['VEi (Total container)'].tolist() #VEi (Số lượng container tàu chở)
VCmi = VesselSchedule['VCmi (Lower limit of quay cranes)'].tolist() #VCmi (Số cần cẩu tối thiểu cho tàu i ) (chiếc)
VCMi = VesselSchedule['VCMi (Upper limit of quay cranes)'].tolist() #VCMi (Số cần cẩu tối đa cho tàu i) (chiếc)
lst1 = [float(x) for x in VesselSchedule['Di1 (Distance vessel i at berth 1 - target \nyard, km)']]
lst2 = [float(x) for x in VesselSchedule['Di2 (Distance vessel i at berth 2 - target \nyard, km)']]
Dij = [[l1, l2] for l1, l2 in zip(lst1, lst2)]
D = np.array([[d[0]] for d in Dij], dtype=float)  # Đảm bảo D là mảng float (V, 1)
B = ChannelBerth['ID bến'].tolist() #Berth
BLj = ChannelBerth['BLj (Chiều dài của bến j ) (mét)'].tolist() #BLj (Chiều dài của bến j ) (mét)
BDj = ChannelBerth['BDj (Mớn nước tại bến j ) (mét)'].tolist() #BDj (Mớn nước tại bến j ) (mét)
Tt = [Tidal.iloc[0, 2], Tidal.iloc[2, 2]] #Khung thoi gian thuy trieu cao
Uiu = False #1 nếu tàu i đi vào kênh ở vị trí u_th khi thủy triều dâng else 0
Viu = False #1 nếu tàu i đi ra khỏi kênh ở phía u_th khi thủy triều dâng else 0
w1 = InputVariables.iloc[15, 2] #trọng số có thể thay đổi tùy theo mục tiêu ưu tiên
w2 = InputVariables.iloc[16, 2] #trọng số có thể thay đổi tùy theo mục tiêu ưu tiên
w3 = InputVariables.iloc[17, 2] #trọng số có thể thay đổi tùy theo mục tiêu ưu tiên
e = InputVariables.iloc[18, 2] #Hằng số logarit
kTA = InputVariables.iloc[19, 2] # Hệ số tính toán thời gian chờ Erlang
l1 = np.random.normal(loc=15.75, scale=1.25) #Tải trọng của xe khi không tải 
l2 = np.random.normal(loc=33.5, scale=1.5) #Tải trọng của xe khi đầy tải 
T = [i for i in range(1,25)]
C = InputVariables.iloc[20, 2] #Crane
Y = [i for i in range(1, 46)] #Truck
xijk = [] #1 nếu tàu i phục vụ theo trình tự k tại bến j else 0
qitn = [] #1 nếu cần cẩu cầu cảng n phục vụ cho tàu i trong thời gian t else 0
u_iu = [] # Biến nhị phân biểu thị khung thời gian tàu có thể vào
v_iu = [] # Biến nhị phân biểu thị khung thời gian tàu có thể ra
VCi = VesselSchedule['VCmi (Lower limit of quay cranes)'].to_list() #Số lượng cần cẩu được phân công cố định cho tàu
VCit = VesselSchedule['VCMi (Upper limit of quay cranes)'].to_list() #Số lượng cần cẩu phục vụ
TWi = [float(val) for val in TWi]
x = []
for i in range(len(TWi)):
    x.append(np.random.normal(loc=float(TWi[i]) / 2, scale=float(TWi[i]) / 2))
    
u_TA = 1.0 / (sum(x) / len(x))
TH1 = [i / vh for i in lst1] #Thời gian xe tải rỗng ddi từ bến 1 
TH2 = [i / vh for i in lst2] #Thời gian xe tải rỗng ddi từ bến 2
TL1 = [i / vl for i in lst1] #Thời gian vận chuyển hàng hóa nặng bằng xe tải từ bến 1
TL2 = [i / vl for i in lst2] #Thời gian vận chuyển hàng hóa nặng bằng xe tải từ bến 2
VmL = [] #nhóm tàu lớn đang cập cảng
CKnt = [] #tập hợp số lượng xe tải phục vụ cẩu tại bến n thời điểm t 
delTA = []  # Độ lệch thời gian dựa trên Erlang
TAi = []    # Thời gian đến thực tế sau khi cộng độ lệch
TE = [] #thời gian khởi hành của tàu đi ra khỏi kênh khi rời cảng
TDTS = []
VO = [] #tập hợp các thứ tự của tàu thuyền vào cảng
VB = [] #Bến tàu i sẽ cập
VC = [] #Số cần cẩu phân bổ cho tàu i
VK = [] #Số xe tải phân bổ cho tàu i
dtb = 500 #Dung tích bình xăng
def calculate_pvl(v, l):
    a = 0.02
    b = -1.67
    c = 0.46
    d = 0.03
    e = 51.17
    return a * v * v + b * v + c * l + d * v * l + e
Dl = float(dtb) / calculate_pvl(vh, l1) * 100 #Tổng quãng đường xe tải đi được khi không có hàng dung tích bình 500l
Dk = float(dtb) / calculate_pvl(vl, l2) * 100#Tổng quãng đường xe tải chở hàng nặng đã đi
tn_ = float(Dl) / vh #Tổng thời gian xe tải chạy không tải tiêu thụ nhiên liệu
def calculate_con13():
    #Sum VCit: tong so can cau phan bo trong suot qua trinh -> = VCMi
    for i in range(len(VEi)):
        TDTS.append(float(VEi[i])/(CE0 * VCit[i]))
calculate_con13()

def convert_time_to_float(time_val):
    if isinstance(time_val, str):
        dt = datetime.strptime(time_val, "%d/%m/%Y %H:%M:%S")
    elif isinstance(time_val, pd.Timestamp) or isinstance(time_val, datetime):
        dt = time_val
    else:
        raise ValueError(f"Unsupported type: {type(time_val)}")
    epoch = datetime(1970, 1, 1)
    delta = dt - epoch
    return delta.total_seconds()
TO_float = [convert_time_to_float(t) for t in TOi]
def calculate_fErlang():
    for i in range(len(x)):
        fxku_ = (pow(u_TA * x[i], kTA - 1) * np.exp(-u_TA * x[i]))
        fxku_ /= math.factorial(kTA - 1)
        delTA.append(fxku_)
    return delTA
def calculate_TA(TO_float, delTAi):
    lst = [TO_float[i] + delTAi[i] for i in range(len(TO_float))]
    return lst
calculate_fErlang()
TAi = calculate_TA(TO_float, delTA)
TF_float = [convert_time_to_float(t) for t in TFi]
for i in range(len(TF_float)):
    TE.append(TF_float[i] - TWi[i])
TAi_float = [float(i) for i in TAi]
TB = [TAi_float[i] + 0.25 for i in range(len(TAi_float))]
TC = [TB[i] + TWi[i] for i in range(len(TB))] #Thời điểm tàu i đã vào cảng
TS = [TC[i] + float(10)/60 for i in range(len(TC))]     #Thời gian bắt đầu công việc cho tàu
TV = TS #Thời gian bắt đầu hỗ trợ từ cần cẩu lân cận cho tàu = TS
TD = [TDTS[i] + TS[i] for i in range(len(TDTS))]
delVCit = [VCMi[i] - VCmi[i] for i in range(len(VCMi))] # Số lượng cần cẩu di chuyển từ bến lân cận để hỗ trợ tàu i tại thời điểm t = VCMi - VCmi
def float_to_datetime(float_time):
    epoch = datetime(1970, 1, 1)
    return epoch + timedelta(seconds=float_time)

np.random.seed(42)
V = len(V_name)   # Số tàu
B = 2   # Số bến
Y = M  # Số xe
Y = 40
M = 99999  # Penalty
TW0 = 24  # Thời gian chờ tối đa (h)
T = 24
CK = np.random.randint(0, 4, size=(C, T))
# Dữ liệu đầu vào

# Thời gian đến (TA) và thời gian xử lý dự kiến (PT)
TA = np.array(TS)  # Giờ
PT = np.array(np.array(TF_float) - np.array(TS))    # Giờ

D = np.array(np.array(Dij))
DMN = 14


# ==================== HÀM NGOÀI CLASS ====================
def get_tide_depth(tide_data, date, hour):
    # Chuyển đổi ngày thành chuỗi nếu cần
    if isinstance(date, datetime):
        date_str = date.strftime('%d/%m/%Y')
    else:
        date_str = str(date)
    
    # Tìm hàng phù hợp với ngày
    row = tide_data[tide_data['Date'].astype(str).str.contains(date_str, na=False)]
    if len(row) == 0:
        return None
    
    # Lấy giá trị độ sâu theo giờ
    hour_col = int(hour) + 1  # Cột bắt đầu từ B (1) đến Y (24)
    return row.iloc[0, hour_col]

# Hàm lấy độ sâu trung bình trong khoảng thời gian
def get_avg_tide_depth(tide_data, date, start_hour, end_hour):
    total = 0
    count = 0
    
    # Lấy giá trị tại các mốc giờ và tính trung bình
    for hour in range(int(start_hour), int(end_hour)):
        depth = get_tide_depth(tide_data, date, hour)
        if depth is not None:
            total += depth
            count += 1
    
    return total / count if count > 0 else None
def get_ship_type(length):
    if length <= 150:
        return 0
    elif length <= 200:
        return 1
    elif length <= 300:
        return 2
    else:
        return 3
def enforce_vessel_berth_constraint(x):
    for i in range(x.shape[0]):
        # Nếu không có bến nào được gán, gán ngẫu nhiên 1 bến
        if np.sum(x[i]) == 0:
            x[i, np.random.randint(x.shape[1])] = 1
        # Nếu có nhiều hơn 1 bến, chỉ giữ lại 1 bến (có thể chọn bến đầu tiên hoặc ngẫu nhiên)
        elif np.sum(x[i]) > 1:
            ones = np.where(x[i] == 1)[0]
            keep = np.random.choice(ones)
            x[i] = 0
            x[i, keep] = 1
    return x
def enforce_crane_constraints(q):
    for i in range(V):
        num_assigned = np.sum(q[i])
        if num_assigned < VCmi[i]:
            available = np.where(q[i] == 0)[0]
            to_add = VCmi[i] - num_assigned
            selected = np.random.choice(available, to_add, replace=False)
            q[i, selected] = 1
        elif num_assigned > VCMi[i]:
            assigned = np.where(q[i] == 1)[0]
            to_remove = num_assigned - VCMi[i]
            selected = np.random.choice(assigned, to_remove, replace=False)
            q[i, selected] = 0
    return q
def schedule_cranes(q2D, VE, CF0, T):
    V, C = q2D.shape
    q3D = np.zeros((V, C, T), dtype=int)

    for i in range(V):  # mỗi tàu
        assigned_cranes = np.where(q2D[i] == 1)[0]
        num_cranes = len(assigned_cranes)
        
        if num_cranes == 0:
            continue

        total_time = int(np.ceil(VE[i] * CF0))  # tính tổng thời gian cần cẩu phục vụ tàu
        duration = total_time // num_cranes
        extra = total_time % num_cranes
        start_time = 0  # có thể đặt tùy theo lịch cập nhật nếu cần

        for j, c in enumerate(assigned_cranes):
            dur = duration + (1 if j < extra else 0)
            end_time = min(start_time + dur, T)
            q3D[i][c][start_time:end_time] = 1
            start_time = end_time  # tiếp nối cần cẩu kế tiếp

    return q3D
def crane_non_crossing_constraint(q):
    for i in range(V):
        for n in range(1, C - 1):
            for t in range(T):
                left = q[i][n - 1][t]
                center = q[i][n][t]
                right = q[i][n + 1][t]
                if left == 1 and right == 1 and center == 0:
                    return False
    return True
def check_truck_constraints(CK, M, D, A, mu_vh, sigma_vh, mu_vl, sigma_vl, CF0, CE0):
    
    C, T = CK.shape
    V, B = D.shape
    
    for t in range(T):
        total_trucks = np.sum(CK[:, t])
        if total_trucks > M:
            return False

    v_h = norm.rvs(mu_vh, sigma_vh)
    v_l = norm.rvs(mu_vl, sigma_vl)

    TH = D / v_h  # Thời gian không tải
    TL = D / v_l  # Thời gian có tải

    # Ràng buộc (37): Hiệu suất cần cẩu không vượt ngưỡng
    for n in range(C):
        for t in range(T):
            i, j = A[n, t]  # Tàu i ở bến j mà cần cẩu n phục vụ tại t
            ck = CK[n, t]

            if ck == 0:
                continue  # Không có xe tải → bỏ qua

            # Tính hiệu suất của cần cẩu
            denominator = max(TH[i, j] + TL[i, j] + 1/CF0, ck / CE0)
            CE_n = ck / denominator

            if CE_n > CE0:
                return False

    return True
def check_vessel_berth_length_constraint(x, VLi, BLj):
    V, B = x.shape
    for i in range(V):
        for j in range(B):
            if x[i][j] == 1 and VLi[i] > BLj[j]:
                return False
    return True
def parse_tide_intervals(Tt):
    intervals = []
    for interval in Tt:
        start_str, end_str = interval.split(" - ")
        h1, m1, s1 = map(int, start_str.split(":"))
        h2, m2, s2 = map(int, end_str.split(":"))
        start = timedelta(hours=h1, minutes=m1, seconds=s1)
        end = timedelta(hours=h2, minutes=m2, seconds=s2)
        intervals.append((start, end))
    return intervals
def is_high_tide(dt, tide_intervals):
    current_time = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
    for start, end in tide_intervals:
        if start <= current_time <= end:
            return True
    return False
def next_high_tide_time(dt, tide_intervals):
    for day_offset in range(0, 3):  # thử tối đa 3 ngày sau
        new_date = dt.date() + timedelta(days=day_offset)
        for start, _ in tide_intervals:
            candidate = datetime.combine(new_date, (datetime.min + start).time())
            if candidate > dt:
                return candidate
    return dt + timedelta(hours=6)  # fallback
def build_CK_A_from_sol(sol, V, C, T, B):
    CK = np.zeros((C, T), dtype=int)
    A = np.zeros((C, T, 2), dtype=int)
    for i in range(V):
        assigned_cranes = np.where(sol['q'][i] == 1)[0]
        assigned_trucks = np.where(sol['y'][i] == 1)[0]
        berth = np.argmax(sol['x'][i])
        for n in assigned_cranes:
            for t in range(T//2):
                CK[n, t] += max(1, len(assigned_trucks)//max(1, len(assigned_cranes)))
                A[n, t] = [i, berth]
    return CK, A
def calc_carbon_emission(sol, V, C, T, D, E1, E2):
    carbon_truck = 0
    carbon_crane = 0
    for i in range(V):
        assigned_cranes = np.where(sol['q'][i] == 1)[0]
        assigned_trucks = np.where(sol['y'][i] == 1)[0]
        berth = np.argmax(sol['x'][i])
        carbon_truck += len(assigned_trucks) * D[i, berth] * 2 * E1
        carbon_crane += len(assigned_cranes) * (T//2) * E2
    return carbon_truck, carbon_crane
def check_berth_safety_constraint(x, VLi, BLj, LSd, LSe):
    V, B = x.shape
    for j in range(B):
        ships = [i for i in range(V) if x[i][j] == 1]
        if not ships:
            continue
        total_length = 0
        total_LSe = 0
        total_LSd = 0
        for idx, i in enumerate(ships):
            ship_type = get_ship_type(VLi[i])
            total_length += VLi[i]
            total_LSe += LSe[ship_type]
            # Khoảng cách giữa các tàu
            if idx > 0:
                prev_ship_type = get_ship_type(VLi[ships[idx-1]])
                # Lấy max khoảng cách giữa 2 tàu liền kề
                total_LSd += max(LSd[ship_type], LSd[prev_ship_type])
        # Nếu chỉ có 1 tàu thì chỉ cộng 1 lần LSd/2
        if len(ships) == 1:
            ship_type = get_ship_type(VLi[ships[0]])
            total_LSd += LSd[ship_type] / 2
        if total_length + total_LSe + total_LSd > BLj[j]:
            return False
    return True

# ==================== THUẬT TOÁN SOA ====================
class SeagullOptimization:
    def __init__(self, n_seagulls=50, max_iter=200):
        self.n_seagulls = n_seagulls
        self.max_iter = max_iter
        
    def calculate_F1(self, TF, TA):
        v = len(TF)
        total = sum(TFi - TAi for TFi, TAi in zip(TF, TA))
        F1 = (1 / v) * total
        return F1
    
    def calculate_F2(self, E1, Dl, Dk, n_, tn_):
        pl = calculate_pvl(vh, l1)
        pk = calculate_pvl(vl, l2)
        F2 = E1 * (pl * Dl + pk * Dk + n_ * tn_)
        return F2
        
    def calculate_F3(self, TD, TS, TV, E2, delVCit, lambda1, lambda2):
        part1 = 0
        part2 = 0
        part3 = 0
        for i in range(len(TD)):
            part1 += (TD[i] - TS[i]) * VCi[i]
            part2 += (TD[i] - TV[i]) * delVCit[i]
            part3 += delVCit[i]
        F3 = E2 * (lambda1 * (part1 + part2) + lambda2 * part3)
        return F3

    def calculate_k123(self, a, b, c):
        return float(1)/a, float(1)/b, float(1)/c

    def calculate_F(self, sol):
        # Tính toán các tham số cần thiết
        F1 = self.calculate_F1(TF_float, TS)
        F2 = self.calculate_F2(E1, Dl, Dk, n_, tn_)
        F3 = self.calculate_F3(TD, TS, TV, E2, delVCit, lambda1, lambda2)
        k1, k2, k3 = self.calculate_k123(F1, F2, F3)
        
        F = w1 * k1 * F1 + w2 * k2 * F2 + w3 * k3 * F3
        return F, F1, F2, F3
    
    def initialize(self):
        self.population = []
        for _ in range(self.n_seagulls):
            sol = {
                'x': np.zeros((V, B), dtype=int),
                'q': np.zeros((V, C), dtype=int),
                'y': np.zeros((V, Y), dtype=int)
            }
            # Phân bổ tàu-bến
            for i in range(V):
                sol['x'][i, np.random.randint(B)] = 1
            
            # Phân bổ cần cẩu
            for i in range(V):
                num_cranes = np.random.randint(VCmi[i], VCMi[i] + 1)
                selected = np.random.choice(C, num_cranes, replace=False)
                sol['q'][i, selected] = 1
            
            # Phân bổ xe
            for i in range(V):
                num_trucks = np.random.randint(1, min(Y, 10) + 1)
                selected = np.random.choice(Y, num_trucks, replace=False)
                sol['y'][i, selected] = 1
            
            self.population.append(sol)
    
    def fitness(self, sol):
        makespan = 0
        penalty = 0
        completion_times = []
        berth_schedule = {j: {'time': 0.0, 'date': None} for j in range(B)}
        tide_intervals = parse_tide_intervals(Tt)

        # Ràng buộc 1: Mỗi tàu chỉ vào 1 bến
        for i in range(V):
            if sum(sol['x'][i]) != 1:
                penalty += 1000 * abs(sum(sol['x'][i]) - 1)

        # Ràng buộc 2: Mỗi cần cẩu chỉ phục vụ 1 tàu tại 1 thời điểm
        for n in range(C):
            if sum(sol['q'][:, n]) > 1:
                penalty += 1000 * (sum(sol['q'][:, n]) - 1)

        for j in range(B):
            # Lấy danh sách các tàu vào bến j
            ships = [i for i in range(V) if sol['x'][i][j] == 1]
            # Sắp xếp các tàu theo thời gian đến
            ships = sorted(ships, key=lambda i: TA[i])

            for idx, i in enumerate(ships):
                berth = j
                assigned_cranes = np.sum(sol['q'][i])
                assigned_trucks = np.sum(sol['y'][i])

                if assigned_cranes == 0 or assigned_trucks == 0:
                    penalty += 10000
                    continue
                
                # Kiểm tra chiều dài tàu không vượt quá chiều dài bến
                if VLi[i] > BLj[berth]:
                    penalty += 10000

                # Tính thời gian phục vụ tàu i
                arrival_dt = float_to_datetime(TA[i])
                arrival_date = arrival_dt.date()
                arrival_hour = arrival_dt.hour + arrival_dt.minute/60

                # Lấy độ sâu tại bến theo giờ
                berth_depth = get_tide_depth(berth_tide, arrival_date, int(arrival_hour))
                if berth_depth is None:
                    # Nếu không có dữ liệu, sử dụng giá trị mặc định
                    berth_depth = BDj[berth]

                # Kiểm tra mớn nước tàu so với độ sâu bến
                if VDi[i] > berth_depth:
                    penalty += 10000

                # Nếu tàu có mớn nước lớn (> DMN), kiểm tra thủy triều
                if VDi[i] > DMN:
                    channel_depth = get_tide_depth(channel_tide, arrival_date, int(arrival_hour))
                    if channel_depth is None:
                        channel_depth = DMN  # Giá trị mặc định nếu không có dữ liệu

                    if VDi[i] > channel_depth:
                        # Tìm thời điểm thủy triều cao tiếp theo
                        next_high_tide = next_high_tide_time(arrival_dt, tide_intervals)
                        TA_adjusted = convert_time_to_float(next_high_tide)
                    else:
                        TA_adjusted = TA[i]
                else:
                    TA_adjusted = TA[i]

                start_time = max(float(TA_adjusted), float(berth_schedule[berth]['time']))
                processing_time = float((VEi[i] * CF0) / assigned_cranes)
                transport_delay = D[i, berth] / (assigned_trucks + 1e-5)
                completion = float(start_time + processing_time + transport_delay)

                # Lưu lại thời gian phục vụ
                sol['start_time'] = sol.get('start_time', np.zeros((V, B)))
                sol['completion'] = sol.get('completion', np.zeros((V, B)))
                sol['start_time'][i, berth] = start_time
                sol['completion'][i, berth] = completion

                # Cập nhật lịch trình bến
                berth_schedule[berth]['time'] = completion
                berth_schedule[berth]['date'] = float_to_datetime(completion).date()

        # Kiểm tra các nhóm tàu cùng lúc ở bến j
        for j in range(B):
            ships = [i for i in range(V) if sol['x'][i][j] == 1]
            for t_idx in range(len(ships)):
                i = ships[t_idx]
                for k_idx in range(t_idx+1, len(ships)):
                    k = ships[k_idx]
                    # Nếu thời gian phục vụ chồng lấn
                    if not (sol['completion'][i, j] <= sol['start_time'][k, j] or 
                           sol['completion'][k, j] <= sol['start_time'][i, j]):
                        # Kiểm tra khoảng cách an toàn
                        overlap_ships = [m for m in ships if not (
                            sol['completion'][m, j] <= sol['start_time'][i, j] or 
                            sol['start_time'][m, j] >= sol['completion'][i, j]
                        )]

                        total_length = 0
                        total_LSe = 0
                        total_LSd = 0

                        for s_idx, m in enumerate(overlap_ships):
                            ship_type = get_ship_type(VLi[m])
                            total_length += VLi[m]
                            total_LSe += LSe[ship_type]
                            if s_idx > 0:
                                prev_ship_type = get_ship_type(VLi[overlap_ships[s_idx-1]])
                                total_LSd += max(LSd[ship_type], LSd[prev_ship_type])

                        if len(overlap_ships) == 1:
                            ship_type = get_ship_type(VLi[overlap_ships[0]])
                            total_LSd += LSd[ship_type] / 2

                        if total_length + total_LSe + total_LSd > BLj[j]:
                            penalty += 10000

        # Các ràng buộc khác giữ nguyên
        q3D = schedule_cranes(sol['q'], VEi, CF0, T)
        if not crane_non_crossing_constraint(q3D):
            penalty += 10000

        if not check_vessel_berth_length_constraint(sol['x'], VLi, BLj):
            penalty += 10000

        CK, A = build_CK_A_from_sol(sol, V, C, T, B)
        mu_vh, sigma_vh = 25, 4
        mu_vl, sigma_vl = 18, 3
        if not check_truck_constraints(CK, M, D, A, mu_vh, sigma_vh, mu_vl, sigma_vl, CF0, E2):
            penalty += 10000

        carbon_truck, carbon_crane = calc_carbon_emission(sol, V, C, T, D, E1, E2)
        w_carbon = 0.01
        makespan = max(completion_times) if completion_times else M
        F, F1, F2, F3 = self.calculate_F(sol)
        return makespan + w_carbon * (carbon_truck + carbon_crane) + penalty + F


    
    def attack_prey(self, best_sol, current_sol):
        new_sol = deepcopy(current_sol)
        for key in ['x', 'q', 'y']:
            mask = np.random.rand(*current_sol[key].shape) < 0.5
            new_sol[key][mask] = best_sol[key][mask]
        new_sol['q'] = enforce_crane_constraints(new_sol['q'])
        new_sol['x'] = enforce_vessel_berth_constraint(new_sol['x'])  # <-- Thêm dòng này
        return new_sol
    
    def migrate(self, sol):
        new_sol = deepcopy(sol)
        for key in ['x', 'y']:
            mutation = np.random.rand(*sol[key].shape) < 0.1
            new_sol[key] = np.logical_xor(sol[key], mutation).astype(int)
        new_sol['x'] = enforce_vessel_berth_constraint(new_sol['x'])  # <-- Thêm dòng này
        # Đột biến có kiểm soát cho cần cẩu
        for i in range(V):
            if np.random.rand() < 0.1:
                num_cranes = np.random.randint(VCmi[i], VCMi[i] + 1)
                new_sol['q'][i] = 0
                selected = np.random.choice(C, num_cranes, replace=False)
                new_sol['q'][i, selected] = 1
        return new_sol
    
    def optimize(self):
        self.initialize()
        best_sol = min(self.population, key=lambda x: self.fitness(x))
        fitness_history = []
        
        for iter in range(self.max_iter):
            for i in range(self.n_seagulls):
                new_sol = self.attack_prey(best_sol, self.population[i])
                if np.random.rand() < 0.3:
                    new_sol = self.migrate(new_sol)
                if self.fitness(new_sol) < self.fitness(self.population[i]):
                    self.population[i] = new_sol
            
            current_best = min(self.population, key=lambda x: self.fitness(x))
            if self.fitness(current_best) < self.fitness(best_sol):
                best_sol = current_best
            
            fitness_history.append(self.fitness(best_sol))
            print(f"Iter {iter+1}, Best Fitness: {fitness_history[-1]}")
        
        plt.plot(fitness_history)
        plt.title("SOA Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness (Makespan + Penalty)")
        plt.grid(True)
        plt.show()
        return best_sol

# ==================== CHẠY THUẬT TOÁN ====================
soa = SeagullOptimization(n_seagulls=30, max_iter=100)
best_solution = soa.optimize()

# ==================== HIỂN THỊ KẾT QUẢ ====================


print("\nBest Solution:")
print("1. Phân bổ tàu-bến (x):")
print(best_solution['x'])

print("\n2. Phân bổ cần cẩu (q):")
print(best_solution['q'])

print("\n3. Phân bổ xe (y):")
print(best_solution['y'])

# Tính toán thời gian chi tiết
berth_schedule = {j: 0 for j in range(B)}
print("\nChi tiết thời gian:")
end_times = []
for i in np.argsort(TA):
    berth = np.argmax(best_solution['x'][i])
    start = max(TS[i], berth_schedule[berth])
    wait = start - TA[i]
    def schedule_cranes(num_containers, num_cranes, cf0=CF0):
        if num_cranes == 0:
            return float('inf')  # Không có cần cẩu thì không thể bốc dỡ
        return num_containers * cf0 / num_cranes

    num_cranes = int(np.sum(best_solution['q'][i])) if 'q' in best_solution else 1
    handling_time = schedule_cranes(VEi[i], num_cranes)
    end = start + handling_time
    end_times.append(end)
    print(f"Tàu {i}: Bến {berth}, Đến {float_to_datetime(TA[i])}h, Bắt đầu {float_to_datetime(start)}h, Chờ {wait / 3600}h, Thời điểm kết thúc dự kiến: {float_to_datetime(end)}h")
    berth_schedule[berth] = start + PT[i]

F, F1, F2, F3 = soa.calculate_F(best_solution)
print("\nCác giá trị mục tiêu:")
print(f"F1 (Thời gian trung bình): {F1} giờ")
print(f"F2 (Phát thải carbon từ xe tải): {F2} kg CO2")
print(f"F3 (Phát thải carbon từ cần cẩu): {F3} kg CO2")
print(f"F (Hàm mục tiêu tổng hợp): {F}")
makespan = (max(end_times) - min(TA)) / 3600  # Đơn vị: giờ
print(f"Makespan thực tế: {makespan} giờ")