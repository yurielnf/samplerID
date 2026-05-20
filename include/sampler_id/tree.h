#pragma once
#include "index_set.h"
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <utility>
#include <stdexcept>
#include <algorithm>

namespace ttid {

using xfac::IndexSet;

// Pure topology: nodes (physical + virtual), ordered neighbors per node, a root.
// Node ids are arbitrary ints; physical leaves are tagged via add_phys().
struct TreeTopology {
    int root = 0;
    std::set<int> phys;                          // physical node ids
    std::map<int, IndexSet<int>> neigh;          // node -> ordered neighbor set

    TreeTopology() = default;
    explicit TreeTopology(int root_) : root(root_) {}

    std::size_t size()   const { return neigh.size(); }
    std::size_t n_phys() const { return phys.size(); }
    bool is_phys(int n)  const { return phys.count(n) > 0; }
    int  deg(int u)      const { return (int)neigh.at(u).size(); }
    int  leg_pos(int u, int v) const { return neigh.at(u).pos(v); }

    void add_edge(int a, int b) {
        if (a == b) throw std::invalid_argument("TreeTopology::add_edge self loop");
        neigh[a].push_back(b);
        neigh[b].push_back(a);
    }
    void add_phys(int n) {
        if (!neigh.count(n)) throw std::invalid_argument("TreeTopology::add_phys unknown node");
        phys.insert(n);
    }

    // Directed edges in root -> leaves order.
    std::vector<std::pair<int,int>> root_to_leaves() const { return root_to_leaves(root); }
    std::vector<std::pair<int,int>> root_to_leaves(int r) const {
        std::vector<std::pair<int,int>> out;
        rtl(out, r, -1);
        return out;
    }

    // Directed edges in leaves -> root order.
    std::vector<std::pair<int,int>> leaves_to_root() const { return leaves_to_root(root); }
    std::vector<std::pair<int,int>> leaves_to_root(int r) const {
        std::vector<std::pair<int,int>> out;
        ltr(out, r, -1);
        return out;
    }

    // Split by removing edge (u,v); return the *physical* nodes on each side.
    std::pair<std::set<int>, std::set<int>> split(int u, int v) const {
        if (!neigh.at(u).to_int().count(v))
            throw std::invalid_argument("TreeTopology::split nodes are not neighbors");
        std::set<int> su, sv;
        collect_side(su, u, v);
        collect_side(sv, v, u);
        std::set<int> pu, pv;
        std::set_intersection(su.begin(), su.end(), phys.begin(), phys.end(),
                              std::inserter(pu, pu.begin()));
        std::set_intersection(sv.begin(), sv.end(), phys.begin(), phys.end(),
                              std::inserter(pv, pv.begin()));
        return {pu, pv};
    }

    void validate() const {
        if (neigh.empty()) return;
        if (!neigh.count(root))
            throw std::invalid_argument("TreeTopology::validate root not in graph");
        std::map<int,int> parent;
        std::set<int> visited;
        std::queue<int> q;
        int start = neigh.begin()->first;
        parent[start] = -1; q.push(start); visited.insert(start);
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            for (int nb : neigh.at(cur).from_int()) {
                if (!visited.count(nb)) { visited.insert(nb); parent[nb] = cur; q.push(nb); }
                else if (nb != parent[cur])
                    throw std::invalid_argument("TreeTopology::validate cycle detected");
            }
        }
        if (visited.size() != neigh.size())
            throw std::invalid_argument("TreeTopology::validate graph is not connected");
    }

    // Build a linear chain topology on n physical nodes (0..n-1), root at 0.
    static TreeTopology chain(int n) {
        TreeTopology t(0);
        for (int i = 0; i < n; ++i) t.neigh[i];
        for (int i = 0; i < n - 1; ++i) t.add_edge(i, i + 1);
        for (int i = 0; i < n; ++i) t.add_phys(i);
        return t;
    }

    friend bool operator==(TreeTopology const& a, TreeTopology const& b) {
        if (a.root != b.root || a.phys != b.phys) return false;
        if (a.neigh.size() != b.neigh.size()) return false;
        for (auto const& [u, ns] : a.neigh) {
            auto it = b.neigh.find(u);
            if (it == b.neigh.end()) return false;
            if (ns.from_int() != it->second.from_int()) return false;
        }
        return true;
    }
    friend bool operator!=(TreeTopology const& a, TreeTopology const& b) { return !(a == b); }

private:
    void rtl(std::vector<std::pair<int,int>>& out, int u, int parent) const {
        for (int w : neigh.at(u).from_int()) {
            if (w == parent) continue;
            out.push_back({u, w});
            rtl(out, w, u);
        }
    }
    void ltr(std::vector<std::pair<int,int>>& out, int u, int parent) const {
        for (int w : neigh.at(u).from_int()) {
            if (w == parent) continue;
            ltr(out, w, u);
            out.push_back({w, u});
        }
    }
    void collect_side(std::set<int>& s, int u, int blocked) const {
        s.insert(u);
        for (int w : neigh.at(u).from_int())
            if (w != blocked && !s.count(w)) collect_side(s, w, blocked);
    }
};

} // namespace ttid
