#include "kmv.hpp"

#include <limits>
#include <stdexcept>

KMinValues::KMinValues(size_t m)
        : m(m)
{
        if (m == 0)
                throw std::invalid_argument("KMinValues: m must be positive");

        k = m / 64;
        if (k < 2)
                throw std::invalid_argument("KMinValues: m must be >= 128");
}

void KMinValues::tryInsertElement(uint64_t hash)
{
        // literally all it does is maintain the smallest k elements
        if (kValSet.count(hash) == 0) {
                if (kValHeap.size() < k) {
                        kValHeap.push(hash);
                        kValSet.insert(hash);
                } else if (hash < kValHeap.top()) {
                        kValSet.insert(hash);
                        kValHeap.push(hash);

                        uint64_t top = kValHeap.top();
                        kValHeap.pop();
                        kValSet.erase(top);
                }
        }
}

// NOTE: x is an already hashed value
void KMinValues::update(uint64_t x)
{
        tryInsertElement(x);
}

std::optional<double> KMinValues::estimate() const
{
        // override for n < k
        if (kValSet.size() < k)
                return kValSet.size();

        // The heap holds k distinct hashes and k >= 2, so its maximum is non-zero.
        uint64_t hash = kValHeap.top();

        // unbiased basic estimator (equiv to D^UB_k in the paper), normalised by the
        // maximum hash value
        return (double)(k - 1) * std::numeric_limits<uint64_t>::max() / hash;
}

void KMinValues::merge(const KMinValues& other)
{
        if (k != other.k)
                throw std::invalid_argument("KMinValues: cannot merge sketches of different k");

        if (this == &other)
                return;

        for (uint64_t hash : other.kValSet)
                tryInsertElement(hash);
}

bool KMinValues::operator==(const KMinValues& other) const
{
        return k == other.k && kValSet == other.kValSet;
}

std::size_t KMinValues::memory_bits() const
{
        return k * 64;
}
