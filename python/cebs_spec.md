# CEBS Format

## Overview


| Offset | Size | Type      | Field                | Notes                                                                  |
|-------:|-----:|-----------|----------------------|------------------------------------------------------------------------|
|      0 |    4 | uint8[4]  | `magic`              | `0x43 0x45 0x42 0x53` = `CEBS`                                         |
|      4 |    4 | uint8[4]  | `version`            | `0x61 0x00 0x00 0x00` = version `a.0.0.0`                              |
|      8 |    8 | uint64_le | `data_start`         | Read as uint32_le                                                      |
|     16 |    8 | uint64_le | `stream_type`        | Read as uint32_le                                                      |
|     24 |    8 | uint64_le | `bytes_per_element`  | Read as uint32_le                                                      |
|     32 |    8 | uint64_le | `number_of_elements` | Read as uint32_le                                                      |
|     40 |   88 | uint8[88] | `reserved`           |                                                                        |



## Stream Type

| Value      | Type    |
|------------|---------|
| `00000000` | uint64  |
| `00000001` | int64   |
| `00000010` | float64 |
| `00001000` | uuid    |
| `11111111` | invalid |


## Version

| Value | `a` | 0  | 0  | 0  |
|-------|-----|----|----|----|
| Hex   | 61  | 00 | 00 | 00 |


## Hex Dump

```
Offset    Hex                                 Field
     0    43 45 42 53                         magic = "CEBS"
     4    61 00 00 00                         version = a.0.0.0
     8    80 00 00 00 00 00 00 00             data_start = 128
    16    00 00 00 00 00 00 00 00             stream_type = uint64
    24    08 00 00 00 00 00 00 00             bytes_per_element = 8
    32    0a 00 00 00 00 00 00 00             number_of_elements = 10
    40    ?? ?? ... ?? (88 bytes)             reserved

   128    00 00 00 00 00 00 00 00             value[0]   = 0
   136    02 00 00 00 00 00 00 00             value[1]   = 2
   144    04 00 00 00 00 00 00 00             value[2]   = 4
   152    06 00 00 00 00 00 00 00             value[3]   = 6
   160    08 00 00 00 00 00 00 00             value[4]   = 8
   168    0a 00 00 00 00 00 00 00             value[5]   = 10
   176    0c 00 00 00 00 00 00 00             value[6]   = 12
   184    0e 00 00 00 00 00 00 00             value[7]   = 14
   192    10 00 00 00 00 00 00 00             value[8]   = 16
   200    12 00 00 00 00 00 00 00             value[9]   = 18
```
