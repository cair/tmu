#ifndef TMU_MEMORY_H
#define TMU_MEMORY_H

#include <vector>
#include <functional>
#include <cstdint>
#include <span>
#include <stdexcept>

template<typename T>
class TMMemory {
public:
    using ReallocateCallback = std::function<void(std::size_t oldCapacity, std::size_t newCapacity)>;

private:
    alignas(64) std::vector<T> memory;
    ReallocateCallback reallocateCallback;
    std::size_t capacity;
    std::size_t cursor = 0; // Cursor to track the current position

public:
    TMMemory() : capacity(0) {}

    TMMemory(const std::vector<T>& data, std::size_t _cursor)
    : memory(data)
    , capacity(data.capacity())
    , cursor(_cursor)
    {}

    void setReallocateCallback(ReallocateCallback callback) {
        reallocateCallback = std::move(callback);
    }

    void reserve(std::size_t newCapacity) {
        if (newCapacity > memory.capacity()) {
            std::size_t oldCapacity = memory.capacity();
            memory.reserve(newCapacity);

            // Fill newly allocated memory with zeros
            memory.resize(newCapacity);
            std::fill(memory.begin() + oldCapacity, memory.end(), 0);

            capacity = memory.capacity();
            if (reallocateCallback) {
                reallocateCallback(oldCapacity, capacity);
            }
        }
    }

    void push_back(T value) {
        if (memory.capacity() == memory.size()) {
            std::size_t oldCapacity = memory.capacity();
            memory.push_back(value);
            std::size_t newCapacity = memory.capacity();
            if (oldCapacity != newCapacity && reallocateCallback) {
                reallocateCallback(oldCapacity, newCapacity);
            }
        } else {
            memory.push_back(value);
        }
    }

    std::span<T> getSegment(std::size_t segmentSize){
        if (cursor + segmentSize > memory.size()) {
            throw std::out_of_range("Segment exceeds memory bounds. Requested '" + std::to_string(cursor + segmentSize) + " bytes' is more than '" + std::to_string(memory.size() - cursor) + "' bytes.");
        }

        std::span<T> segmentView(&memory[cursor], segmentSize);
        cursor += segmentSize; // Advance the cursor
        return segmentView;
    }

    // Resets the cursor to the beginning or to a specified position
    void resetCursor(std::size_t newCursor = 0) {
        if (newCursor > memory.size()) {
            throw std::out_of_range("New cursor position exceeds memory bounds.");
        }
        cursor = newCursor;
    }

};

#endif //TMU_MEMORY_H
