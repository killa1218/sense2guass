import re

reg = re.compile('^(\w+)\t(\w+)\t(\d+\.\d+).*$')

fp = open('LEDS_result')
# fp = open('BLESS_result')
# fp = open('Medical_result')
# fp = open('TM14_result')

total = 0
count = 0

lines = fp.readlines()

for line in lines:
    total += 1
    if float(reg.match(line).group(3)) > 0.49:
        count += 1


print 'total: ' + str(total)
print 'count: ' + str(count)
print float(count) / total
