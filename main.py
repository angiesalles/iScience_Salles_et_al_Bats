import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wknml
import scipy.interpolate
import scipy.io as sio
import os.path
from collections import defaultdict
import platform
import plotly.express as px
from sklearn.cluster import KMeans
import math
import pickle
import flask
import sys
sys.path.append('/home/angie/capybara/')
from definitions import todos
use_webknossos = True
import scipy.stats

#%%

def plot_fun(fn):
    assert(fn.endswith('.png'))
    plt.savefig(fn.replace('.png','.pdf'))

lookup = ['e', 'f', 'a', 'g', 'm', 's']

path = '/Volumes/angie/Efuscus/hungerGames/' 
computer = platform.uname()
if computer.system == 'Windows':
    path = 'data/'
if computer.system == 'Linux':
    path = '/home/angie/Efuscus/hungerGames/'
beh_collect = defaultdict(list)
beh_collect2 = defaultdict(list)
beh_collect_per_trial = list()
beh_collect2_per_trial = list()
batty_collect = list()
batty_rms_collect = list()
mov_collect = list()
loudness_collect = list()
pertrial_collect = list()
pertrial_collect_dots = list()
behname_vs_callname = defaultdict(list)
pairing_vs_callname = defaultdict(list)
pairing_vs_callnumber = defaultdict(list)
pairing_vs_behavior_time = defaultdict(list)
pairing_vs_behavior = defaultdict(list)
collector = []
done_pairing_counter = []
for trial in todos:
    beh_collect_per_trial.append(defaultdict(list))
    beh_collect2_per_trial.append(defaultdict(list))
    path2 = path + 'AnalyzedAudio/'
    pertrial_collect.append(defaultdict(list))
    batty_collect.append(defaultdict(list))
    pertrial_collect_dots.append(defaultdict(list))
    loudness_collect.append(defaultdict(list))
    mov_collect.append(defaultdict(list))
    batty_rms_collect.append(defaultdict(list))
    audio_data = pd.read_csv(path2 + trial[0],
                             header=None,
                             usecols=[4, 5],
                             names=['times_s', 'end_s'])
    audiopath = path + 'MicWavfiles/' + trial[0][:8] + '/Pair_Trial1_' + trial[0][9:-4] + '.wav'
    if os.path.exists(audiopath + '.pickle'):
        with open(audiopath + '.pickle', 'rb') as pfile:
            batty_data = pickle.load(pfile)['labels']
        
        if len(batty_data) > 0 and 'type_call' not in batty_data[-1]:# file not done yet
            batty_data = []
        else:
            done_pairing_counter.append(trial[4])
    else:
        batty_data = []
    raw_samplerate, raw_audio_data = sio.wavfile.read(audiopath)
    path3 = path + 'AnalyzedVideo/'
    behavior_data = pd.read_csv(path3 + trial[0][:9]+trial[1]+'_bat1.dat',
                                comment='#',
                                skiprows=20,
                                header=None,
                                names=['time_old_ms', 'label'],
                                sep=',',
                                skipinitialspace=True)
    behavior_data2 = pd.read_csv(path3 + trial[0][:9]+trial[1]+'_bat2.dat',
                                comment='#',
                                skiprows=20,
                                header=None,
                                names=['time_old_ms', 'label'],
                                sep=',',
                                skipinitialspace=True)
    merge_todo = [['r','m'], # moving and retreating are both moving and not significantly different in call rate of spike rate
                  ['c','g']] # "cuddling" to "grooming"
    for mt in merge_todo:
        for behav_todo in [behavior_data,behavior_data2]:
            for labelin in range(len(behav_todo['label'])):
                if behav_todo['label'][labelin] == mt[0]:
                    
                    behav_todo.loc[labelin, 'label'] = mt[1]
    
    
    if use_webknossos:
        nml_list = os.listdir(path + 'NML')
        nml_name = None
        for item in nml_list:
            if item.startswith(trial[0].split('_')[0] + '_' + trial[1]):
                nml_name = item
        with open(path + 'NML' + os.sep + nml_name + os.sep + nml_name + '.nml') as f:
            nml = wknml.parse_nml(f)
        tree_names = [tree.name for tree in nml.trees]


        def get_nodes(tree_name):
            tree = nml.trees[tree_names.index(tree_name)]
            return [node.position for node in tree.nodes]


        nodes_here = get_nodes(tree_names[0])
        assert(len(nodes_here) > 20)
        nodes_here2 = []
        for idx in range(3):
            nodes_here2.append([c[idx] for c in nodes_here])
        fx = scipy.interpolate.interp1d(nodes_here2[2], nodes_here2[0])
        fy = scipy.interpolate.interp1d(nodes_here2[2], nodes_here2[1])
        xcoords = fx(range(int(max(nodes_here2[2]))))
        ycoords = fy(range(int(max(nodes_here2[2]))))
    else:
        with open(path
                  + 'DeepLabCut_output/hungergames1-angie-2022-06-12/videos/'
                  + trial[0][:8] + '_' + trial[1]
                  + 'DLC_dlcrnetms5_hungerGames1Jun12shuffle1_200000_full.pickle', 'rb') as f:
            dlc_data = pickle.load(f)
        del dlc_data['metadata']
        assert (min([len(item['coordinates']) for item in dlc_data.values()]) == 1)
        assert (max([len(item['coordinates']) for item in dlc_data.values()]) == 1)
        assert (all([key.startswith('frame') for key in dlc_data]))
        xcoords = []
        ycoords = []
        for item in dlc_data.values():
            if len(item['coordinates'][0][2]) == 0:
                xcoords.append(lastgoodx)
                ycoords.append(lastgoody)
            else:
                lastgoodx = item['coordinates'][0][2][0][0]
                lastgoody = item['coordinates'][0][2][0][1]
                xcoords.append(item['coordinates'][0][2][0][0])
                ycoords.append(item['coordinates'][0][2][0][1])
    behavior_data['start_s'] = behavior_data['time_old_ms']/1000 * 10.0 / (9.0+50.0/60.0)
    behavior_data['end_s'] = np.hstack([behavior_data['start_s'][1:], [np.nan]])
    behavior_data['duration'] = behavior_data['end_s'] - behavior_data['start_s']
    behavior_data2['start_s'] = behavior_data2['time_old_ms']/1000 * 10.0 / (9.0+50.0/60.0)
    behavior_data2['end_s'] = np.hstack([behavior_data2['start_s'][1:], [np.nan]])
    behavior_data2['duration'] = behavior_data2['end_s'] - behavior_data2['start_s']
    pairing_vs_behavior[trial[4]].append(behavior_data['label'])
    pairing_vs_behavior_time[trial[4]].append(behavior_data['duration'])
    
    collector.append(behavior_data)
    path4 = path + '/NeuralData/' + trial[0][:8] + '_Matfile/' + trial[2] + '/'
    offset_data = sio.loadmat(path4+'Chn17.mat')
    offset = np.argwhere(offset_data['data'] > 5)[5, 1]/offset_data['sr'][0, 0]
    matstorage = []
    for neuro_idx in range(16):
        neuro_path = path4 + 'times_Chn'+str(neuro_idx+1)+'.mat'
        if not os.path.exists(neuro_path):
            continue
        neuro_data = sio.loadmat(neuro_path)['cluster_class']
        matstorage.append(neuro_data)
        del neuro_data
    pairing_vs_callnumber[trial[4]].append(len(audio_data['times_s']))
    for idx in range(len(audio_data['times_s'])):
        item = audio_data['times_s'][idx]
        if trial[3] == 'Elvy' and item > 120:
            #print(item)
            continue

        def my_rms(x):
            return np.sqrt(np.mean((x - x.mean()) ** 2))

        res_my_rms = my_rms(raw_audio_data[int(audio_data['times_s'][idx] * raw_samplerate):int(
            audio_data['end_s'][idx] * raw_samplerate)])
        if idx < len(batty_data):
            batty_key = batty_data[idx]['type_call']
        else:
            batty_key = 'nix'
        pairing_vs_callname[batty_key].append(trial[4])
        batty_rms_collect[-1][batty_key].append(res_my_rms)

        beh_idx = np.argwhere(behavior_data['start_s'].to_numpy() < item)

        if beh_idx.size > 0:

            keytemp = behavior_data['label'][beh_idx[-1]].to_numpy()[0]
            pertrial_collect[-1][keytemp].append(0)
            batty_collect[-1][batty_key].append(0)
            
            loudness_collect[-1][keytemp].append(res_my_rms)
            behname_vs_callname[batty_key].append(keytemp)

            for neuro_data_idx in range(len(matstorage)):
                neuro_data = matstorage[neuro_data_idx]
                good_times = neuro_data[neuro_data[:, 0] > 0, :]
                good_times[:, 1] /= 1000
                good_times[:, 1] -= offset
                search_radius = 0.025
                if idx < len(audio_data['times_s'])-1:
                    search_radius = min(search_radius, audio_data['times_s'][idx+1]-item)
                good_times2 = good_times.copy()
                good_times2 = good_times2[good_times2[:, 1] > item, :]
                good_times2 = good_times2[good_times2[:, 1] < item+0.025, :]

                valuetemp = good_times2[:, 0].size*0.025/search_radius
                pertrial_collect[-1][keytemp][-1] += valuetemp
                good_times2[:,1] -= item
                good_times2[:,0] += 100 * neuro_data_idx
                pertrial_collect_dots[-1][batty_key].append(good_times2)
                batty_collect[-1][batty_key][-1] += valuetemp
    for idx in range(behavior_data.shape[0]):
        key = behavior_data['label'][idx]
        beh_collect[key].append(behavior_data['duration'][idx])
        beh_collect_per_trial[-1][key].append(behavior_data['duration'][idx])
        if math.isnan(behavior_data['start_s'][idx]) or math.isnan(behavior_data['end_s'][idx]):
            continue
        limit_left = int(behavior_data['start_s'][idx] * len(xcoords) // (60 * 10))
        limit_right = int(behavior_data['end_s'][idx] * len(xcoords) // (60 * 10))
        mov_collect[-1][key].append(np.sum(np.sqrt(np.diff(xcoords[limit_left:limit_right])**2
                                                   + np.diff(ycoords[limit_left:limit_right])**2)))
    for idx in range(behavior_data2.shape[0]):
        key = behavior_data2['label'][idx]
        beh_collect2[key].append(behavior_data2['duration'][idx])
        beh_collect2_per_trial[-1][key].append(behavior_data2['duration'][idx])                                       


color_map = {'a': 'deeppink',
             'e': 'green',
             'f': 'orange',
             'g': 'saddlebrown',
             'm': 'purple',
             'r': 'purple',
             's': 'grey',
             'c': 'gold',
             'EOF': 'white'}

batty_keys = np.setdiff1d([item for mydict in batty_rms_collect for item in mydict.keys()],
                             ['nix','on','Unclear','Unsure'])

sexlist = ['FM', 'MM', 'FF']
#%% oldfig 1
plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)


stats_save=list()
for idx in range(len(lookup)):
    stats_save.append([item2 for item in pertrial_collect for item2 in item[lookup[idx]]])
    data_temp = np.array([np.nanmean(item[lookup[idx]]) for item in pertrial_collect])
    data_temp = data_temp[~np.isnan(data_temp)]
    bplot = ax.boxplot(data_temp, 
                       positions=[idx],
                       patch_artist=True,
                       medianprops = dict(color="black"))
    boxhere = bplot['boxes'][0]
    boxhere.set_facecolor(color_map[lookup[idx]])
    boxhere.set_alpha(0.5)
print(scipy.stats.kruskal(*stats_save))
plt.xticks(ticks=range(len(lookup)), labels=lookup)
plt.xlabel('behavior')
plt.ylabel('mean evoked spike number per call')


plot_fun('mean_evoked_firing.png')

#%% oldfig 2
plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)
data_temp_col = []

for idx in range(len(batty_keys)):
    key = batty_keys[idx]
    data_temp = np.array([np.nanmean(item[key]) for item in batty_collect])
    data_temp = data_temp[~np.isnan(data_temp)]
    bplot = ax.boxplot(data_temp,
                       patch_artist=True,
                       positions=[idx],
                       medianprops = dict(color="black"))
    boxhere = bplot['boxes'][0]
    boxhere.set_alpha(0.5)
    data_temp_col.append(data_temp)
plt.xticks(ticks=range(len(batty_keys)), labels=batty_keys)
plt.xlabel('call type')
plt.ylabel('mean evoked spike number per call')

plt.xticks(rotation=70)
print(np.mean(sum([list(item) for item in data_temp_col],[])))
plot_fun('batty_evoked_firing.png')
#%%
for idx in range(13):
    print(scipy.stats.ranksums(data_temp_col[idx],sum([list(x) for x in np.array(data_temp_col, dtype=object)[np.setdiff1d(list(range(13)),[idx])]],[])))

#%% oldfig 3

plt.figure()
for idx in range(len(lookup)):
    data_temp = np.zeros([0, 1])
    for trial in pertrial_collect:
        data_temp = np.append(data_temp, len(trial[lookup[idx]]))
    plt.errorbar(idx,
                 data_temp.mean(),
                 data_temp.std()/np.sqrt(len(data_temp)),
                 fmt='o',
                 color=color_map[lookup[idx]],
                 ecolor='black')

plt.xticks(ticks=range(len(lookup)), labels=lookup)
plt.xlabel('behavior')
plt.ylabel('calls per trial')
plot_fun('calls_per_trial.png')

#%% oldfig 4

try:
    del beh_collect['EOF']
except:
    pass
try:
    del beh_collect['t']
except:
    pass

totz = sum([sum(x) for x in beh_collect.values()])

for key in beh_collect:
    print(key)
    print(sum(beh_collect[key])*100/totz)

plt.figure()
mylabs = beh_collect.keys()
piecolors = [color_map[key] for key in mylabs]
plt.pie([sum(x) for x in beh_collect.values()], labels=mylabs, colors=piecolors)
plot_fun('some_pie.png')

#%% oldfig 5


plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)


for idx, key in enumerate(lookup):
    data_temp = np.array([np.nanmean(item[key]) for item in mov_collect])
    data_temp = data_temp[~np.isnan(data_temp)]
    
    bplot = ax.boxplot(data_temp, 
                       positions=[idx],
                       patch_artist=True,
                       medianprops = dict(color="black"))
    boxhere = bplot['boxes'][0]
    boxhere.set_facecolor(color_map[lookup[idx]])
    boxhere.set_alpha(0.5)

plt.ylabel('distance traveled [a.u.]')
plt.xticks(ticks=range(len(lookup)), labels=lookup)
plt.xlabel('behavior')
plot_fun('distance_traveled.png')




#%% oldfig 6


plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)

for idx, key in enumerate(lookup):
    data_temp = np.array([np.nanmean(item[key]) for item in loudness_collect])
    data_temp = data_temp[~np.isnan(data_temp)]
    bplot = ax.boxplot(data_temp,
                       patch_artist=True,
                       positions=[idx],
                       medianprops = dict(color="black"))
    boxhere = bplot['boxes'][0]
    boxhere.set_alpha(0.5)
plt.ylabel('loudness')
plt.xticks(ticks=range(len(lookup)), labels=lookup)
plt.xlabel('behavior')
plot_fun('loudness.png')

plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)


for idx, key in enumerate(batty_keys):
    data_temp = np.array([np.nanmean(item[key]) for item in batty_rms_collect])
    data_temp = data_temp[~np.isnan(data_temp)]
    print(key)
    print(np.std(data_temp))
    bplot = ax.boxplot(data_temp,
                       patch_artist=True,
                       positions=[idx],
                       medianprops = dict(color="black"))
    boxhere = bplot['boxes'][0]
    boxhere.set_alpha(0.5)
plt.xticks(ticks=range(len(batty_keys)), labels=batty_keys)
plt.xlabel('call type')
plt.ylabel('loudness')

plt.xticks(rotation=70)

plot_fun('loudness_batty.png')

#%% oldfig 7

from sklearn.metrics import r2_score



plt.figure(figsize=(8,6))
x_store=[]
y_store=[]


for idx, key in enumerate(batty_keys):
    data_temp_y = np.array([np.nanmean(item[key]) for item in batty_collect])
    data_temp_x = np.array([np.nanmean(item[key]) for item in batty_rms_collect])
    x_store.append(np.nanmean(data_temp_x)/1000)
    y_store.append(np.nanmean(data_temp_y))
    plt.scatter(x_store[-1],y_store[-1],label=key,marker='o^'[idx//10])
z = np.polyfit(x_store, y_store, 1)
p = np.poly1d(z)
coefficient_of_dermination = r2_score(y_store, p(x_store))

#add trendline to plot
rmserror= np.sqrt(np.mean((y_store-p(x_store))**2))
plt.plot(x_store, p(x_store),'k',label=f'y={z[0].round(3)}x+{z[1].round(2)}\nRMS={rmserror.round(2)}\nRsquared={coefficient_of_dermination.round(2)}')

#plt.xticks(ticks=range(len(batty_collect)), labels=batty_collect.keys())
#plt.xlabel('type call')
#plt.ylabel('mean evoked firing rate per call type normalized loudness')
plt.legend(loc='lower right')
plt.xlabel('RMS [a.u.]')
plt.ylabel('mean evoked spike number per call')
plot_fun('loudness_vs_calltype.png')
#%% oldfig 8

batty_rms_collect[lidx]

plt.figure()
for idx in range(len(batty_collect)):
    data_temp = np.array(batty_collect[list(batty_collect.keys())[idx]])
    plt.scatter(idx, np.nanmean(batty_rms_collect[lidx]/data_temp),np.nanstd(batty_rms_collect[lidx]/data_temp), color=color_map[lookup[idx]])

plt.xticks(ticks=range(len(batty_collect)), labels=batty_collect.keys())
plt.xlabel('type call')
plt.ylabel('mean evoked firing rate per call type normalized loudness')

plot_fun('batty_evoked_firing_normalized_loudness.png')

#%% item 1
data_temp_col = dict()
for idx in range(len(lookup)):
    data_temp = np.zeros([0, 1])
    for trial in pertrial_collect:
        data_temp = np.append(data_temp, len(trial[lookup[idx]]))
    data_temp_col[lookup[idx]] = data_temp
print(scipy.stats.kruskal(*[data_temp_col[lookup[idx]] for idx in range(len(lookup))]))

selectitem = 's'
for idx in range(len(lookup)):
    if lookup[idx] == selectitem:
        continue
    print(lookup[idx])
    print(scipy.stats.ranksums(data_temp_col[selectitem],data_temp_col[lookup[idx]]))

#%% item 2
allcollect = defaultdict(list)
for idx, item in enumerate(pertrial_collect):
    for key in lookup:
        allcollect[key]+=item[key] 

print(scipy.stats.kruskal(*[allcollect[lookup[idx]] for idx in range(len(lookup))]))

for idx in range(len(lookup)):
    if lookup[idx] == 'a':
        continue
    print(lookup[idx])
    print(scipy.stats.ranksums(allcollect['a'],allcollect[lookup[idx]]))

#%% item 3

#print(scipy.stats.kruskal(*[np.array(mov_collect[lookup[idx]]) for idx in range(len(lookup))]))

for key in lookup:
    if key == 's':
        continue
    print(key)
    a=sum([item['s'] for item in mov_collect],[])
    b=sum([item[key] for item in mov_collect],[])
    print(scipy.stats.ranksums(a,b)) #even significant with bonferoni for all except grooming


#%% item 4
loudness_collect_full = defaultdict(list)
for idx, item in enumerate(loudness_collect):
    for key in lookup:
        loudness_collect_full[key]+=item[key] 

print(scipy.stats.kruskal(*[np.array(loudness_collect_full[lookup[idx]]) for idx in range(len(lookup))]))
for idx in range(len(lookup)):
    if lookup[idx] == 'e':
        continue
    print(lookup[idx])
    print(scipy.stats.ranksums(loudness_collect_full['e'],loudness_collect_full[lookup[idx]]))

#%% item 5
batty_rms_collect_full = defaultdict(list)
for idx, item in enumerate(batty_rms_collect):
    for key in batty_keys:
        batty_rms_collect_full[key]+=item[key] 

print(scipy.stats.kruskal(*[batty_rms_collect_full[list(batty_rms_collect_full.keys())[idx]] for idx in range(len(batty_rms_collect_full))]))

#%% item 6
batty_collect_full = defaultdict(list)
for idx, item in enumerate(batty_collect):
    for key in batty_keys:
        batty_collect_full[key]+=item[key] 

print(scipy.stats.kruskal(*[batty_collect_full[list(batty_collect_full.keys())[idx]] for idx in range(len(batty_collect_full))]))


#%% item 7 (oldfig 9)
types_of_call = batty_keys
for type_of_call in types_of_call:
    collector = np.zeros([100,0])
    for trial in range(len(todos)):
        tempx = []
        for item2 in pertrial_collect_dots[trial][type_of_call]:
            for item3 in item2:
                tempx.append(item3[0])
        num_idx = len(np.unique(np.setdiff1d(tempx,[0])))
        if num_idx  == 0:
            continue
        collector_pre = np.zeros([100, num_idx])
        for item2 in pertrial_collect_dots[trial][type_of_call]:
            for item3 in item2:
                if item3[0]%100 > 0:
                    collector_pre[int(item3[1]*4000),list(np.unique(np.setdiff1d(tempx,[0]))).index(item3[0])] += 1
        collector = np.hstack([collector, collector_pre])
        assert(collector.shape[0]==100)
    #plt.figure()
    #model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    
    #y_pred = KMeans(n_clusters=2,).fit_predict(collector)
    #plt.imshow(collector[y_pred, :].T)
    #plt.title(type_of_call)
    #plot_fun('neurons_' + type_of_call + '.png')
#%%
collectora=defaultdict(int)
collectora_echo=defaultdict(int)
collectora_per_call=defaultdict(int)
collectora_echo_per_call=defaultdict(int)

collectorb=defaultdict(int)
traffic =defaultdict(int)
traffic_echo = defaultdict(int)
collector_per_call_global=defaultdict(lambda: defaultdict(int))
collector_global=defaultdict(lambda: defaultdict(int))
for idx, item in enumerate(pertrial_collect_dots):
    print(idx)
    for item2 in item['FMB']:
        for item3 in item2:
            collectora[str(idx)+'_' +str(item3[0])]+=1
            collectora_per_call[str(idx)+'_' +str(item3[0])]+=1/len(item['FMB'])
            traffic[str(idx)+'_' +str(item3[0])]=len(item['FMB']) # this is a unique neuron identifier
    for item2 in item['Echo']:
        for item3 in item2:
            collectora_echo[str(idx)+'_' +str(item3[0])]+=1 # this is a unique neuron identifier
            collectora_echo_per_call[str(idx)+'_' +str(item3[0])]+=1/len(item['Echo']) # this is a unique neuron identifier
            traffic_echo[str(idx)+'_' +str(item3[0])]=len(item['Echo']) 
    for key in item:
        for item2 in item[key]:
            for item3 in item2:
                collector_per_call_global[key][str(idx)+'_' +str(item3[0])]+=1/len(item['Echo']) # this is a unique neuron identifier
                collector_global[key][str(idx)+'_' +str(item3[0])]+=1

    for key in item:
        for item2 in item[key]:
            for item3 in item2:
                collectorb[str(idx)+'_'+str(item3[0])]+=1
#%%
collectorc = {}
for key in collectora:
    collectorc[key]=collectora[key]/collectorb[key]
#%%
plt.figure()
plt.scatter(collectora.values(),collectorc.values())
plt.xlabel('number of spikes during FMB calls')
plt.ylabel('fraction of spikes during FMB calls')
plot_fun('item7.png')
#%%
col_x=[]
col_y=[]
plt.figure()
for key in np.union1d(list(collectora_echo_per_call.keys()),list(collectora_per_call.keys())):
    if traffic[key]>0 and traffic_echo[key]>9:
        col_x.append(collectora_echo_per_call[key])
        col_y.append(collectora_per_call[key])
plt.scatter(col_x,col_y)
plt.xlabel('spikes per echo call')
plt.ylabel('spikes per FMB')

#%%
allcalls=np.setdiff1d(list(collector_per_call_global.keys()),['nix','Unsure','Unclear','on'])
allneurons = np.unique(sum([list(collector_per_call_global[x].keys()) for x in allcalls],[]))
heatmap = np.zeros([len(allcalls),len(allneurons)])
heatmap_naive = heatmap.copy()
for [idx1, item1] in enumerate(allcalls):
    for idx2, item2 in enumerate(allneurons):
        heatmap[idx1,idx2] = collector_per_call_global[item1][item2]
        heatmap_naive[idx1,idx2] = collector_global[item1][item2]
plt.figure()
plt.imshow(heatmap[:,:]) #FMB 422 #LDFM 660
#%%

heatmapnorm=heatmap/np.sum(heatmap, axis=0)
heatmapnorm=heatmapnorm[[2,1,3,12,4,5,6,0,7,8,9,10,11],:]
leadercolumn = np.argmax(heatmapnorm, axis=0)
heatmapnorm2 = heatmapnorm.copy()
for idx, item in enumerate(leadercolumn):
    heatmapnorm2[item, idx] = 0
leadercolumn2 = np.argmax(heatmapnorm2, axis=0)
sorted_indices = np.lexsort((leadercolumn2, leadercolumn))

plt.figure(figsize=(18, 16))
plt.imshow(np.repeat(np.log(heatmapnorm[:,sorted_indices]+0.01),30,axis=0))
plt.yticks(np.array(range(13))*30+15, labels=allcalls[[2,1,3,12,4,5,6,0,7,8,9,10,11]])
plot_fun('forangie.png')
#%%
from PIL import Image
Image.fromarray(heatmapnorm[:,sorted_indices]).save('forangieraw.tif')

#%% item 8
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

iris = load_iris()
X = iris.data

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
ZZ= plot_dendrogram(model, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
#%% item 9 (oldfig 10)
collect_here = []
for key1 in lookup:
    for key2 in behname_vs_callname:
        if key2 != 'nix':
            collect_here.append([key1, key2, behname_vs_callname[key2].count(key1)])
        
        
df = pd.DataFrame(collect_here,columns=['group','column','val'])

df.head()
fig = px.bar(df, y="column", x="val",
             color='group', barmode='group',text="column",
             height=400)
fig.write_image('behname_vs_callname.pdf')
for key2 in behname_vs_callname:
    print(key2)
    print(sum([item[2] for item in collect_here if item[1]==key2 and item[0]=='a'])/sum([item[2] for item in collect_here if item[1]==key2])*100)

heatstore = []
heatstore_legend = []
collect_here = []
for key2 in behname_vs_callname:
    if key2 != 'nix' and key2 != 'on' and key2 != 'Unsure' and key2 != 'Unclear':
        heatstore.append([])
        heatstore_legend.append(key2)
        for key1 in lookup:

    
            heatstore[-1].append(behname_vs_callname[key2].count(key1))
            collect_here.append([key2, key1, np.log(1+behname_vs_callname[key2].count(key1)/len(behname_vs_callname[key2]))])
    
        
df = pd.DataFrame(collect_here,columns=['group','column','val'])

df.head()
fig = px.bar(df, y="column", x="val",
             color='group', barmode='group',text="column",
             height=400)
fig.write_image('behname_vs_callname_reverse.pdf')

plt.figure()
plt.imshow(np.log(np.array(heatstore).T+1))
plt.yticks(list(range(len(lookup))),labels=lookup)
plt.xticks(list(range(len(heatstore_legend))),labels=heatstore_legend,rotation=90)
plot_fun('call_vs_behavior_heatmap.png')
for idx in range(13):
    print(heatstore_legend[idx])
    print(np.sum(np.array(heatstore)[idx,:])*100/np.sum(np.array(heatstore)))
    
#%% item 10

#%% item 11 (oldfig 11)
heatmap = []
collect_here = []
heatmap_simple = []
collect_here_simple =[]
sorted_callnames = ['Echo', 'DFM', 'CS',  'FMB', 'QCF', 'SFM', 'QFC-DFM', 'sHFM', 'LDFM', 'LFM', 
                    'LsDFM-LFM',  'LQCF-CS', 'UFM']
appeasement = ['sHFM', 'LDFM', 'LFM', 'LsDFM-LFM',  'LQCF-CS', 'UFM']
for key1 in ['FM', 'MM', 'FF']:
    heatmap.append([])
    heatmap_simple.append([0,0])
    for key2 in sorted_callnames:
        if key2 != 'nix' and key2 != 'on' and key2 != 'Unsure' and key2 != 'Unclear':
            heatmap[-1].append(pairing_vs_callname[key2].count(key1)/done_pairing_counter.count(key1))
            collect_here.append([key1, 
                                 key2, 
                                 pairing_vs_callname[key2].count(key1)/done_pairing_counter.count(key1)])
            if key2 == 'Echo':
                continue
            if key2  in appeasement:
                heatmap_simple[-1][0]+=pairing_vs_callname[key2].count(key1)/done_pairing_counter.count(key1)
            else:
                heatmap_simple[-1][1]+=pairing_vs_callname[key2].count(key1)/done_pairing_counter.count(key1)
for idx1,key1 in enumerate(sexlist):
    for idx2 in range(2):
        collect_here_simple.append([key1,['Appeasement','Aggression'][idx2],heatmap_simple[idx1][idx2]/np.sum(heatmap_simple,axis=1)[idx1]])
df = pd.DataFrame(collect_here, columns=['group', 'column', 'val'])

df.head()
fig = px.bar(df, y="column", x="val",
             color='group', barmode='group', text="column",
             height=400)

fig.write_image('pairing_vs_callname.pdf')
df = pd.DataFrame(collect_here_simple, columns=['group', 'column', 'val'])

df.head()
fig = px.bar(df, y="val", x="column",
             color='group', barmode='group', text="group",
             height=400)

fig.write_image('pairing_vs_stateofmind.pdf')
plt.figure()
heatmap = np.array(heatmap/np.sum(heatmap, axis=0))
plt.imshow(heatmap)
plt.yticks([0,1,2],labels=sexlist)
plt.xticks(list(range(len(sorted_callnames))),labels=sorted_callnames,rotation=90)
plot_fun('pairing_vs_callname_heatmap.png')

plt.figure()

for idx in sexlist:
    plt.errorbar(idx,
                 np.array(pairing_vs_callnumber[idx]).mean(),
                 np.array(pairing_vs_callnumber[idx]).std()/np.sqrt(len(pairing_vs_callnumber)),
                 fmt='o',
                 ecolor='black')
plt.title('pairing vs call number')
plot_fun('pairing_vs_callnumber.png')


collect_here = []
collect_heatmap = np.zeros([len(sexlist),len(lookup)])
for idx1, key1 in enumerate(sexlist):
    for idx2, key2 in enumerate(lookup):
        counter = 0
        for item in pairing_vs_behavior[key1]:
            counter+= list(item).count(key2)
        actual_number = counter/[trial[4] for trial in todos].count(key1)
        collect_here.append([key1, key2, actual_number])
        collect_heatmap[idx1,idx2] = actual_number
        

df = pd.DataFrame(collect_here, columns=['group', 'column', 'val'])

df.head()
fig = px.bar(df, y="column", x="val",
             color='group', barmode='group', text="column",
             height=400)

fig.write_image('pairing_vs_behavior.pdf')

plt.figure()
plt.imshow(collect_heatmap/np.tile(collect_heatmap.sum(axis=1),[6,1]).T)
plt.yticks([0,1,2],labels=sexlist)
plt.xticks(list(range(len(lookup))),labels=lookup)
plot_fun('pairing_vs_behavior_heatmap.png')

#%% finality 1
collect_heatmap_time = np.zeros([len(sexlist),len(lookup)])

for lookup_idx, key in enumerate(lookup):
    print(key)
    collect_here = defaultdict(list)
    for sexlist_idx, key1 in enumerate(sexlist):
        for idx,item in enumerate(pairing_vs_behavior[key1]):
            spentona = 0
            spentall = 0
            for idx2,item2 in enumerate(item):
                if not item2 == 'EOF':
                    spentall+=pairing_vs_behavior_time[key1][idx][idx2]
                if item2 == key:
                    spentona+= pairing_vs_behavior_time[key1][idx][idx2]
                    collect_heatmap_time[sexlist_idx,lookup_idx] += pairing_vs_behavior_time[key1][idx][idx2]
            collect_here[key1].append(spentona/spentall)
            
    print(scipy.stats.kruskal(*[collect_here[key1] for key1 in ['FM', 'MM', 'FF']]))        

plt.figure()
plt.imshow(collect_heatmap_time/np.tile(collect_heatmap_time.sum(axis=1),[6,1]).T)
plt.yticks([0,1,2],labels=sexlist)
plt.xticks(list(range(len(lookup))),labels=lookup)
plot_fun('pairing_vs_behavior_heatmap_time.png')


#%% finality 2
for key in lookup:
     print(key)
     #remember bonferroni
     print(scipy.stats.ranksums([len(item[key]) for item in beh_collect_per_trial],
                                [len(item[key]) for item in beh_collect2_per_trial]))
fig, ax = plt.subplots(nrows=1, ncols=1)
for idx, key in enumerate(lookup):
    bplot = ax.boxplot([len(item[key]) for item in beh_collect_per_trial], 
                       positions=[idx-0.15],
                       patch_artist=True,
                       medianprops = dict(color="black"))
    boxhere = bplot['boxes'][0]
    boxhere.set_facecolor(color_map[lookup[idx]])
    boxhere.set_alpha(0.5)
    bplot = ax.boxplot([len(item[key]) for item in beh_collect2_per_trial], 
                       positions=[idx+0.15],
                       patch_artist=True,
                       medianprops = dict(color="black"))
    boxhere = bplot['boxes'][0]
    boxhere.set_facecolor(color_map[lookup[idx]])
    boxhere.set_alpha(0.5)
    boxhere.set(hatch = '///')
plt.xticks(ticks=range(len(lookup)), labels=lookup)
plt.xlabel('behavior')
plt.ylabel('occurences per trial')

plt.xticks(rotation=70)

plot_fun('beh_occrs_inplntd_vs_not.png')
#%% finality 3
collect_here = defaultdict(int)
for item in batty_collect:
    for key in item:
        collect_here[key]+=len(item[key])
del collect_here['Unsure']
for key in collect_here:
    print(key)
    print(collect_here[key]/sum(collect_here.values()))
plt.figure()        
plt.pie(collect_here.values(),labels=collect_here.keys())
plot_fun('finality3.png')


