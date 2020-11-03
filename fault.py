import os
import random
import time

cmd_cpu = './blade create cpu load --cpu-count %d --timeout %d'
cmd_mem = './blade c mem load --mode ram --mem-percent %d --timeout %d'
cmd_disk_read = './blade create disk burn --read --path /home --timeout %d'
cmd_disk_write = './blade create disk burn --write --path /home --timeout %d'
cmd_net_in = 'iperf3 -c 222.201.144.198 -t %d -R'
cmd_net_out = 'iperf3 -c 222.201.144.198 -t %d'


while True:
    rate = random.randint(1, 100)
    sleep = random.randint(40, 60)
    time.sleep(sleep)
    if rate < 3:
        continue
    cpu_timeout = random.randint(5, 25)
    cpu_rate = random.randint(36, 80)
    mem_rate = random.randint(50, 90)
    net_rate = random.randint(459, 951)

    timestamp = time.time()
    case = random.randint(1, 4)
    f = open('/root/location.txt', 'a+')

    if case == 1:
        os.system(cmd_cpu%(cpu_rate, cpu_timeout))
        type_error = 'CPU injection time in:%d and consist in:%d'
        f.write(type_error%(timestamp, cpu_timeout) + '\n')
        f.close()

    elif case == 2:
        os.system(cmd_mem%(mem_rate, cpu_timeout))
        type_error = 'MEM injection time in:%d and consist in:%d'
        f.write(type_error % (timestamp, cpu_timeout) + '\n')
        f.close()


    elif case == 3:
        ToF = random.randint(0,1)
        if(ToF == 0):
            os.system(cmd_disk_read%(cpu_timeout))
            type_error = 'DISK_READ injection time in:%d and consist in:%d'
            f.write(type_error % (timestamp, cpu_timeout) + '\n')
            f.close()
        else:
            os.system(cmd_disk_write%(cpu_timeout))
            type_error = 'DISK_WRITE injection time in:%d and consist in:%d'
            f.write(type_error % (timestamp, cpu_timeout) + '\n')
            f.close()


    elif case == 4:
        ToF = random.randint(0,1)
        if(ToF == 0):
            os.system(cmd_net_in%(cpu_timeout))
            type_error = 'NET_IN injection time in:%d and consist in:%d'
            f.write(type_error % (timestamp, cpu_timeout) + '\n')
            f.close()
        else:
            os.system(cmd_net_out%(cpu_timeout))
            type_error = 'NET_OUT injection time in:%d and consist in:%d'
            f.write(type_error % (timestamp, cpu_timeout) + '\n')
            f.close()
