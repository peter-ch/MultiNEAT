#ifndef MULTINEAT_SERIALIZATION_H
#define MULTINEAT_SERIALIZATION_H

///////////////////////////////////////////////////////////////////////////////////////////
//    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Serialization.h
// Description: Minimal dependency-free archives (a drop-in replacement for the
//              subset of boost::serialization that MultiNEAT used).
//
//              Classes opt in by providing
//                  template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/)
//              and writing "ar & member;" for every member, exactly like boost.
//
//              The format is a whitespace separated token stream that is independent
//              of the current locale. Doubles are stored as their exact bit pattern,
//              so values (including NaN/inf) round-trip without loss.
///////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cstring>
#include <istream>
#include <locale>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifdef PYTHON_BINDINGS
#include <pybind11/pybind11.h>
#endif

namespace NEAT
{

class OArchive
{
public:
    static constexpr bool is_loading = false;

    explicit OArchive(std::ostream &a_os) : m_os(a_os)
    {
        m_os.imbue(std::locale::classic());
    }

    template<class T>
    OArchive &operator&(const T &a_v)
    {
        save(a_v);
        return *this;
    }

    template<class T>
    OArchive &operator<<(const T &a_v)
    {
        save(a_v);
        return *this;
    }

private:
    std::ostream &m_os;

    template<class T>
    void save(const T &a_v)
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            m_os << (a_v ? 1 : 0) << ' ';
        }
        else if constexpr (std::is_enum_v<T>)
        {
            m_os << static_cast<long long>(a_v) << ' ';
        }
        else if constexpr (std::is_integral_v<T>)
        {
            m_os << static_cast<long long>(a_v) << ' ';
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            double d = static_cast<double>(a_v);
            std::uint64_t bits;
            std::memcpy(&bits, &d, sizeof(bits));
            m_os << bits << ' ';
        }
        else
        {
            const_cast<T &>(a_v).serialize(*this, 0);
        }
    }

    void save(const std::string &a_s)
    {
        m_os << a_s.size() << ':';
        m_os.write(a_s.data(), static_cast<std::streamsize>(a_s.size()));
        m_os << ' ';
    }

    template<class T, class A>
    void save(const std::vector<T, A> &a_v)
    {
        m_os << a_v.size() << ' ';
        for (const auto &x : a_v)
        {
            save(x);
        }
    }

    template<class K, class V, class C, class A>
    void save(const std::map<K, V, C, A> &a_m)
    {
        m_os << a_m.size() << ' ';
        for (const auto &kv : a_m)
        {
            save(kv.first);
            save(kv.second);
        }
    }

    template<class... Ts>
    void save(const std::variant<Ts...> &a_v)
    {
        m_os << a_v.index() << ' ';
        std::visit([this](const auto &x) { this->save(x); }, a_v);
    }

#ifdef PYTHON_BINDINGS
    // Python objects are stored with the standard pickle module
    void save(const pybind11::object &a_o)
    {
        if (!a_o || a_o.is_none())
        {
            m_os << 0 << ' ';
            return;
        }
        m_os << 1 << ' ';
        pybind11::bytes b = pybind11::module_::import("pickle").attr("dumps")(a_o, -1);
        save(static_cast<std::string>(b));
    }
#endif
};


class IArchive
{
public:
    static constexpr bool is_loading = true;

    explicit IArchive(std::istream &a_is) : m_is(a_is)
    {
        m_is.imbue(std::locale::classic());
    }

    template<class T>
    IArchive &operator&(T &a_v)
    {
        load(a_v);
        return *this;
    }

    template<class T>
    IArchive &operator>>(T &a_v)
    {
        load(a_v);
        return *this;
    }

private:
    std::istream &m_is;

    void check()
    {
        if (m_is.fail())
        {
            throw std::runtime_error("MultiNEAT: corrupt or incompatible serialized data");
        }
    }

    std::size_t load_size()
    {
        unsigned long long n = 0;
        m_is >> n;
        check();
        return static_cast<std::size_t>(n);
    }

    template<class T>
    void load(T &a_v)
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            int x = 0;
            m_is >> x;
            check();
            a_v = (x != 0);
        }
        else if constexpr (std::is_enum_v<T> || std::is_integral_v<T>)
        {
            long long x = 0;
            m_is >> x;
            check();
            a_v = static_cast<T>(x);
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            std::uint64_t bits = 0;
            m_is >> bits;
            check();
            double d;
            std::memcpy(&d, &bits, sizeof(d));
            a_v = static_cast<T>(d);
        }
        else
        {
            a_v.serialize(*this, 0);
        }
    }

    void load(std::string &a_s)
    {
        std::size_t n = load_size();
        char colon = 0;
        m_is.get(colon);
        if (colon != ':')
        {
            throw std::runtime_error("MultiNEAT: corrupt or incompatible serialized data");
        }
        a_s.resize(n);
        if (n > 0)
        {
            m_is.read(&a_s[0], static_cast<std::streamsize>(n));
        }
        check();
    }

    template<class T, class A>
    void load(std::vector<T, A> &a_v)
    {
        std::size_t n = load_size();
        a_v.clear();
        a_v.resize(n);
        for (std::size_t i = 0; i < n; i++)
        {
            load(a_v[i]);
        }
    }

    template<class K, class V, class C, class A>
    void load(std::map<K, V, C, A> &a_m)
    {
        std::size_t n = load_size();
        a_m.clear();
        for (std::size_t i = 0; i < n; i++)
        {
            K k{};
            load(k);
            V v{};
            load(v);
            a_m[k] = std::move(v);
        }
    }

    template<class V, std::size_t I = 0>
    void load_variant_alternative(V &a_v, std::size_t a_idx)
    {
        if constexpr (I < std::variant_size_v<V>)
        {
            if (a_idx == I)
            {
                std::variant_alternative_t<I, V> x{};
                load(x);
                a_v = std::move(x);
            }
            else
            {
                load_variant_alternative<V, I + 1>(a_v, a_idx);
            }
        }
        else
        {
            throw std::runtime_error("MultiNEAT: corrupt or incompatible serialized data (variant index)");
        }
    }

    template<class... Ts>
    void load(std::variant<Ts...> &a_v)
    {
        std::size_t idx = load_size();
        load_variant_alternative(a_v, idx);
    }

#ifdef PYTHON_BINDINGS
    void load(pybind11::object &a_o)
    {
        int present = 0;
        m_is >> present;
        check();
        if (!present)
        {
            a_o = pybind11::none();
            return;
        }
        std::string s;
        load(s);
        a_o = pybind11::module_::import("pickle").attr("loads")(pybind11::bytes(s));
    }
#endif
};


// Convenience helpers
template<class T>
std::string SerializeToString(const T &a_obj)
{
    std::ostringstream os;
    OArchive oa(os);
    oa << a_obj;
    return os.str();
}

template<class T>
void DeserializeFromString(const std::string &a_data, T &a_obj)
{
    std::istringstream is(a_data);
    IArchive ia(is);
    ia >> a_obj;
}

} // namespace NEAT

#endif // MULTINEAT_SERIALIZATION_H
