#!/usr/bin/python3
import subprocess


def main():
    # for nb in ('0', '1'):
    for nb in (str(x) for x in (0, 1)):
        for measure in map(str, range(1,3)):
            l = []
            for num_threads in range(1, 17):
                print("nb {}, measure {}, num_threads {}".format(nb, measure, num_threads))
                print("Compiling...")
                p = subprocess.Popen(["make", "MEASURE=" + measure, "NON_BLOCKING=" + nb,
                    "NB_THREADS={}".format(num_threads)], stdout=subprocess.PIPE)
                p.wait()
                print("Running...")
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
            print("DONE WITH {m}_{nb}".format(m=measure, nb=nb))
            with open("shit_{m}_{nb}.csv".format(m=measure, nb=nb), 'w') as f:
                for x in l:
                    f.write(str(x) + '\n')


if __name__ == '__main__':
    main()

