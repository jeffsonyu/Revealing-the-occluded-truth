/*
 CircularBuffer.h - Circular buffer library for Arduino.
 Copyright (c) 2017 Roberto Lo Giacco.

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as 
 published by the Free Software Foundation, either version 3 of the 
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CIRCULAR_BUFFER_H_
#define CIRCULAR_BUFFER_H_
#include <stdint.h>
#include <stddef.h>

#ifdef CIRCULAR_BUFFER_DEBUG
#include <Print.h>
#endif

namespace Helper {
	/** @private */
	template<bool FITS8, bool FITS16> struct Index {
		using Type = uint32_t;
	};

	/** @private */
	template<> struct Index<false, true> {
		using Type = uint16_t;
	};

	/** @private */
	template<> struct Index<true, true> {
		using Type = uint8_t;
	};
}

/**
 * @brief Implements a circular buffer that supports LIFO and FIFO operations.
 *
 * @tparam T The type of the data to store in the buffer.
 * @tparam S The maximum number of elements that can be stored in the buffer.
 * @tparam IT The data type of the index. Typically should be left as default.
 */
template<typename T, size_t S, typename IT = typename Helper::Index<(S <= UINT8_MAX), (S <= UINT16_MAX)>::Type> class CircularBuffer {
public:
	/**
	 * @brief The buffer capacity.
	 *
	 * Read only as it cannot ever change.
	 */
	static constexpr IT capacity = static_cast<IT>(S);

	/**
	 * @brief Aliases the index type.
	 *
	 * Can be used to obtain the right index type with `decltype(buffer)::index_t`.
	 */
	using index_t = IT;

	/**
	 * @brief Create an empty circular buffer.
	 */
	constexpr CircularBuffer();

	// disable the copy constructor
	/** @private */
	CircularBuffer(const CircularBuffer&) = delete;
	/** @private */
	CircularBuffer(CircularBuffer&&) = delete;

	// disable the assignment operator
	/** @private */
	CircularBuffer& operator=(const CircularBuffer&) = delete;
	/** @private */
	CircularBuffer& operator=(CircularBuffer&&) = delete;

	/**
	 * @brief Adds an element to the beginning of buffer.
	 *
	 * @return `false` iff the addition caused overwriting to an existing element.
	 */
	bool unshift(T value);

	/**
	 * @brief Adds an element to the end of buffer.
	 *
	 * @return `false` iff the addition caused overwriting to an existing element.
	 */
	bool push(T value);

	/**
	 * @brief Removes an element from the beginning of the buffer.
	 *
	 * @warning Calling this operation on an empty buffer has an unpredictable behaviour.
	 */
	T shift();

	/**
	 * @brief Removes an element from the end of the buffer.
	 *
	 * @warning Calling this operation on an empty buffer has an unpredictable behaviour.
	 */
	T pop();

	/**
	 * @brief Returns the element at the beginning of the buffer.
	 *
	 * @return The element at the beginning of the buffer.
	 */
	T inline first() const;

	/**
	 * @brief Returns the element at the end of the buffer.
	 *
	 * @return The element at the end of the buffer.
	 */
	T inline last() const;

	/**
	 * @brief Array-like access to buffer.
	 *
	 * Calling this operation using and index value greater than `size - 1` returns the tail element.
	 *
	 * @warning Calling this operation on an empty buffer has an unpredictable behaviour.
	 */
	T operator [] (IT index) const;

	/**
	 * @brief Returns how many elements are actually stored in the buffer.
	 *
	 * @return The number of elements stored in the buffer.
	 */
	IT inline size() const;

	/**
	 * @brief Returns how many elements can be safely pushed into the buffer.
	 *
	 * @return The number of elements that can be safely pushed into the buffer.
	 */
	IT inline available() const;

	/**
	 * @brief Check if the buffer is empty.
	 *
	 * @return `true` iff no elements can be removed from the buffer.
	 */
	bool inline isEmpty() const;

	/**
	 * @brief Check if the buffer is full.
	 *
	 * @return `true` if no elements can be added to the buffer without overwriting existing elements.
	 */
	bool inline isFull() const;

	/**
	 * @brief Resets the buffer to a clean status, making all buffer positions available.
	 *
	 * @note This does not clean up any dynamically allocated memory stored in the buffer.
	 * Clearing a buffer that points to heap-allocated memory may cause a memory leak, if it's not properly cleaned up.
	 */
	void inline clear();

	#ifdef CIRCULAR_BUFFER_DEBUG
	void inline debug(Print* out);
	void inline debugFn(Print* out, void (*printFunction)(Print*, T));
	#endif

private:
	T buffer[S];
	T *head;
	T *tail;
#ifndef CIRCULAR_BUFFER_INT_SAFE
	IT count;
#else
	volatile IT count;
#endif
};

/*
 CircularBuffer.tpp - Circular buffer library for Arduino.
 Copyright (c) 2017 Roberto Lo Giacco.

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as 
 published by the Free Software Foundation, either version 3 of the 
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

template<typename T, size_t S, typename IT>
constexpr CircularBuffer<T,S,IT>::CircularBuffer() :
		head(buffer), tail(buffer), count(0) {
}

template<typename T, size_t S, typename IT>
bool CircularBuffer<T,S,IT>::unshift(T value) {
	if (head == buffer) {
		head = buffer + capacity;
	}
	*--head = value;
	if (count == capacity) {
		if (tail-- == buffer) {
			tail = buffer + capacity - 1;
		}
		return false;
	} else {
		if (count++ == 0) {
			tail = head;
		}
		return true;
	}
}

template<typename T, size_t S, typename IT>
bool CircularBuffer<T,S,IT>::push(T value) {
	if (++tail == buffer + capacity) {
		tail = buffer;
	}
	*tail = value;
	if (count == capacity) {
		if (++head == buffer + capacity) {
			head = buffer;
		}
		return false;
	} else {
		if (count++ == 0) {
			head = tail;
		}
		return true;
	}
}

template<typename T, size_t S, typename IT>
T CircularBuffer<T,S,IT>::shift() {
	if (count == 0) return *head;
	T result = *head++;
	if (head >= buffer + capacity) {
		head = buffer;
	}
	count--;
	return result;
}

template<typename T, size_t S, typename IT>
T CircularBuffer<T,S,IT>::pop() {
	if (count == 0) return *tail;
	T result = *tail--;
	if (tail < buffer) {
		tail = buffer + capacity - 1;
	}
	count--;
	return result;
}

template<typename T, size_t S, typename IT>
T inline CircularBuffer<T,S,IT>::first() const {
	return *head;
}

template<typename T, size_t S, typename IT>
T inline CircularBuffer<T,S,IT>::last() const {
	return *tail;
}

template<typename T, size_t S, typename IT>
T CircularBuffer<T,S,IT>::operator [](IT index) const {
	if (index >= count) return *tail;
	return *(buffer + ((head - buffer + index) % capacity));
}

template<typename T, size_t S, typename IT>
IT inline CircularBuffer<T,S,IT>::size() const {
	return count;
}

template<typename T, size_t S, typename IT>
IT inline CircularBuffer<T,S,IT>::available() const {
	return capacity - count;
}

template<typename T, size_t S, typename IT>
bool inline CircularBuffer<T,S,IT>::isEmpty() const {
	return count == 0;
}

template<typename T, size_t S, typename IT>
bool inline CircularBuffer<T,S,IT>::isFull() const {
	return count == capacity;
}

template<typename T, size_t S, typename IT>
void inline CircularBuffer<T,S,IT>::clear() {
	head = tail = buffer;
	count = 0;
}

#ifdef CIRCULAR_BUFFER_DEBUG
#include <string.h>
template<typename T, size_t S, typename IT>
void inline CircularBuffer<T,S,IT>::debug(Print* out) {
	for (IT i = 0; i < capacity; i++) {
		int hex = (int)buffer + i;
		out->print("[");
		out->print(hex, HEX);
		out->print("] ");
		out->print(*(buffer + i));
		if (head == buffer + i) {
			out->print("<-head");
		} 
		if (tail == buffer + i) {
			out->print("<-tail");
		}
		out->println();
	}
}

template<typename T, size_t S, typename IT>
void inline CircularBuffer<T,S,IT>::debugFn(Print* out, void (*printFunction)(Print*, T)) {
	for (IT i = 0; i < capacity; i++) {
		int hex = (int)buffer + i;
		out->print("[");
		out->print(hex, HEX);
		out->print("] ");
		printFunction(out, *(buffer + i));
		if (head == buffer + i) {
			out->print("<-head");
		} 
		if (tail == buffer + i) {
			out->print("<-tail");
		}
		out->println();
	}
}
#endif

#endif
