#include <stdint.h>
#include <stdlib.h>

typedef struct {
    void *data;
    uint64_t length;
} Array;

Array generate_range(uint64_t start, uint64_t stop, uint64_t step) {
    if (step == 0 || stop < start) exit(1);

    uint64_t count = (stop - start) / step + ((stop - start) % step != 0);
    uint64_t *range = malloc(count * sizeof(*range));
    for (uint64_t i = 0; i < count; i++) {
        range[i] = start + i * step;
    }
    return (Array){range, count};
}

void free_range(Array range) {
    free(range.data);
}