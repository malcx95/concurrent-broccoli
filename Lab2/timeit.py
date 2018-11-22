import subprocess


def main():
    l = []
    res = []
    for num_threads in range(1, 17):
        p = subprocess.Popen(["make", "MEASURE=1", "NON_BLOCKING=1",
            "NB_THREADS={}".format(num_threads)], stdout=subprocess.PIPE)
        p.wait()
        out = subprocess.check_output(
            "./stack".format(num_threads)
        ).decode("UTF-8")

        # Split into lines
        lines = out.split("\n");


        time = 0
        for line in lines:
            try:
                time += float(line.split(":")[1].strip())
            except:
                pass
        l.append(time / num_threads)
    with open("shit2.csv", 'w') as f:
        for x in l:
            f.write(str(x) + '\n')


if __name__ == '__main__':
    main()

