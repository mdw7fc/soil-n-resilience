import requests, time, math, json, os
import numpy as np
from collections import defaultdict
LOC = {
 'north_america':   ('US Corn Belt',      41.6, -93.6),
 'europe':          ('NW Europe',         51.0,   3.7),
 'east_asia':       ('N China Plain',     36.5, 116.0),
 'south_asia':      ('Indo-Gangetic',     26.8,  80.9),
 'southeast_asia':  ('SE Asia rice',      15.0, 100.5),
 'latin_america':   ('Cerrado/Pampas',   -16.7, -49.3),
 'sub_saharan_africa':('E/W Africa',        8.5,   4.5),
 'fsu_central_asia':('Ukraine/S Russia',  49.0,  36.2),
}
def ra_mj(lat_rad, doy):
    sd=0.409*math.sin(2*math.pi/365*doy-1.39)
    dr=1+0.033*math.cos(2*math.pi/365*doy)
    x=-math.tan(lat_rad)*math.tan(sd); x=max(-1,min(1,x)); ws=math.acos(x)
    Ra=(24*60/math.pi)*0.0820*dr*(ws*math.sin(lat_rad)*math.sin(sd)+math.cos(lat_rad)*math.cos(sd)*math.sin(ws))
    return max(0.0,Ra)
def fetch(k,lat,lon):
    raw=f'/tmp/era5_raw_{k}.json'
    if os.path.exists(raw):
        return json.load(open(raw))['daily']
    r=requests.get("https://archive-api.open-meteo.com/v1/archive",
       params={"latitude":lat,"longitude":lon,"start_date":"2001-01-01","end_date":"2020-12-31",
       "daily":"temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum",
       "timezone":"UTC"}, timeout=110)
    r.raise_for_status(); j=r.json(); json.dump(j,open(raw,'w')); return j['daily']
def normals(k,lat,lon):
    d=fetch(k,lat,lon); lat_rad=math.radians(lat)
    times=d['time']; tmean=d['temperature_2m_mean']; tmax=d['temperature_2m_max']; tmin=d['temperature_2m_min']; prcp=d['precipitation_sum']
    ymP=defaultdict(float); ymE=defaultdict(float); ymT=defaultdict(list)
    for i,t in enumerate(times):
        y=int(t[:4]); mo=int(t[5:7]); tm=tmean[i]; tx=tmax[i]; tn=tmin[i]; pr=prcp[i]
        if None in (tm,tx,tn,pr): continue
        doy=(np.datetime64(t)-np.datetime64(f'{y}-01-01')).astype(int)+1
        Ra=ra_mj(lat_rad,doy)
        pet=0.0023*(tm+17.8)*math.sqrt(max(0.0,tx-tn))*(0.408*Ra)
        ymP[(y,mo)]+=pr; ymE[(y,mo)]+=max(0.0,pet); ymT[(y,mo)].append(tm)
    tt=[];pp=[];ee=[]
    for mo in range(1,13):
        tt.append(round(float(np.mean([np.mean(ymT[(y,mo)]) for y in range(2001,2021)])),1))
        pp.append(round(float(np.mean([ymP[(y,mo)] for y in range(2001,2021)])),0))
        ee.append(round(float(np.mean([ymE[(y,mo)] for y in range(2001,2021)])),0))
    return tt,pp,ee
out={}
for k,(name,lat,lon) in LOC.items():
    for a in range(3):
        try:
            tt,pp,ee=normals(k,lat,lon); out[k]={'name':name,'lat':lat,'lon':lon,'temp':tt,'precip':pp,'pet':ee}
            print(k,"OK"); break
        except Exception as e:
            print(k,"retry",a,type(e).__name__,str(e)[:100]); time.sleep(4)
json.dump(out,open('/tmp/era5_regional_climates.json','w'),indent=1)
print("saved",len(out))
