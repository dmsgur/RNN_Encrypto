from datetime import datetime
def cv_format(d):#* datetime >> stringformat
    fstr = "%y년 %m월 %d일 %H시 %M분 %S초"
    return d.strftime(fstr)

def cv_mill2date(millsec):#* millsecond >> datetime
    return datetime.fromtimestamp(millsec)

def cv_date2milli(d):# datetime >> millsecond
    return d.timestamp()

def cv_str2date(dstr):#* stringformat >> datetime
    fstr = "%y년 %m월 %d일 %H시 %M분 %S초"
    return datetime.strptime(dstr,fstr)

def gapCompare(start_mill,end_mill):# 두 날자간의 일 시 분 초 리턴
    gap = end_mill - start_mill  # 초
    gap_day = 60 * 60 * 24
    gap_hour = 60 * 60
    gap_min = 60
    gapday = gap // gap_day
    gaphour = (gap - gapday * gap_day) // gap_hour
    gapmin = ((gap - gapday * gap_day) - (gaphour * gap_hour)) // gap_min
    gapsec = (gap - gapday * gap_day) - (gaphour * gap_hour) - (gapmin * gap_min)
    return {"gday":gapday,"ghour":gaphour,"gmin":gapmin,"gsec":gapsec}
#함수 작동 거증
# tdata = 1388070000000/1000
# d = cv_mill2date(tdata)
# dstr=(cv_format(d))
# d=(cv_str2date(dstr))
# dmill = (cv_date2milli(d))
# print(gapCompare(dmill,datetime.now().timestamp()))
if __name__=="__main__":
    pass