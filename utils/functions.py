from utils.constants import metric_names
import re


def parse_num_list(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start), int(end)+1))


def write_metrics(tag, metrics, writer, step):
    # ipc, misses, llc, mbl, mbr, latency = metrics
    header = '{}/'.format(tag)
    for metric, metric_name in zip(metrics, metric_names):
        if metric is not None:
            writer.add_scalar(header + metric_name, metric, step)
    # if tag == LC_TAG:
    #     writer.add_scalar(header + 'Latency', latency, step)
    #
    # writer.add_scalar(header + 'IPC', ipc, step)
    # writer.add_scalar(header + 'Misses', misses, step)
    # writer.add_scalar(header + 'LLC', llc, step)
    # writer.add_scalar(header + 'MBL', mbl, step)
    # writer.add_scalar(header + 'MBR', mbr, step)
    writer.flush()

# use to log latency with this
# latency_per = np.percentile(latency_list, 99)
# latency_list_per = [min(i, latency_per) for i in latency_list]
# plt.plot(latency_list_per)
# plt.title('Effect of collocation in tail latency')
# plt.axvline(x=self.warm_up, color='g', linestyle='dashed', label='BEs starts')
# plt.axvline(x=len(latency_list_per) - self.warm_up, color='r', linestyle='dashed', label='BEs stops')
# plt.axhline(y=self.latency_thr, color='m', label='Latency threshold')
# plt.xlabel('Steps')
# plt.ylabel('Q95 Latency in ms')
# plt.legend(loc='best')
# plt.savefig('runs/collocation_{}.png'.format(datetime.today().strftime('%Y%m%d_%H%M%S')))
# plt.show()
