def schedule_jobs(jobs):

    #Sort jobs by longest time spent in the finishing step, i.e. by decreasing f
    jobs.sort(key=lambda x: x[1], reverse=True)

    current_time = 0
    finish_times = []

    for p, f in jobs:
        current_time += p
        finish_time = current_time + f
        finish_times.append(finish_time)
    
    return max(finish_times)

jobs=[(2,5), (3,5), (7,9)]

jobs1=[(5,2), (5,3), (9,7)]

jobs3=[(5,2), (5,3), (9,1)]

jobs4=[(10,2), (5,3), (1,9)]


print(schedule_jobs(jobs))
print(schedule_jobs(jobs1))
print(schedule_jobs(jobs3))
print(schedule_jobs(jobs4))
