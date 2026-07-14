#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace {
constexpr int N = 4;
constexpr int ACTIONS = 16;
constexpr int CELLS = 64;

struct State {
    std::array<int8_t, CELLS> board{};
    std::array<int8_t, ACTIONS> heights{};
    int8_t to_play = 1;
    int ply = 0;
};

struct Node;
struct Edge {
    int action;
    double prior;
    int visits = 0;
    double value_sum = 0.0;
    std::unique_ptr<Node> child;
};
struct Node { std::vector<Edge> edges; };
struct History { State state; std::array<double, ACTIONS> policy{}; };
struct Game { State state; std::unique_ptr<Node> root = std::make_unique<Node>(); std::vector<History> history; };
struct Leaf { Game* game; State state; Node* node; std::vector<Edge*> path; };

int idx(int x, int y, int z) { return z * 16 + y * 4 + x; }

int winner(const State& s) {
    static constexpr int dirs[13][3] = {
        {1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,-1,0},{1,0,1},{1,0,-1},
        {0,1,1},{0,1,-1},{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1}};
    for (int z=0; z<N; ++z) for (int y=0; y<N; ++y) for (int x=0; x<N; ++x) {
        const int8_t p = s.board[idx(x,y,z)];
        if (!p) continue;
        for (auto& d : dirs) {
            int ex=x+3*d[0], ey=y+3*d[1], ez=z+3*d[2];
            if (ex<0||ex>=N||ey<0||ey>=N||ez<0||ez>=N) continue;
            bool ok=true;
            for (int k=1;k<4;++k) if (s.board[idx(x+k*d[0],y+k*d[1],z+k*d[2])] != p) { ok=false; break; }
            if (ok) return p;
        }
    }
    return 0;
}
bool terminal(const State& s) { return s.ply >= CELLS || winner(s) != 0; }
double terminal_value(const State& s) { int w=winner(s); return w==0 ? 0.0 : (w==s.to_play ? 1.0 : -1.0); }
State play(const State& s, int a) { State n=s; n.board[n.heights[a]*16+a]=s.to_play; ++n.heights[a]; n.to_play=-s.to_play; ++n.ply; return n; }

PyObject* state_to_python(const State& s, PyObject* state_cls) {
    PyObject* board=PyTuple_New(CELLS), *heights=PyTuple_New(ACTIONS);
    if (!board || !heights) { Py_XDECREF(board); Py_XDECREF(heights); return nullptr; }
    for(int i=0;i<CELLS;++i) PyTuple_SET_ITEM(board,i,PyLong_FromLong(s.board[i]));
    for(int i=0;i<ACTIONS;++i) PyTuple_SET_ITEM(heights,i,PyLong_FromLong(s.heights[i]));
    PyObject* obj=PyObject_CallFunction(state_cls,"OOii",board,heights,(int)s.to_play,s.ply);
    Py_DECREF(board); Py_DECREF(heights); return obj;
}

double rng_random(PyObject* rng) {
    PyObject* r=PyObject_CallMethod(rng,"random",nullptr); if(!r) return -1.0;
    double v=PyFloat_AsDouble(r); Py_DECREF(r); return v;
}
double rng_gamma(PyObject* rng, double alpha) {
    PyObject* r=PyObject_CallMethod(rng,"gammavariate","dd",alpha,1.0); if(!r) return -1.0;
    double v=PyFloat_AsDouble(r); Py_DECREF(r); return v;
}

Edge* select_edge(Node& n, double c) {
    int total=0; for(auto& e:n.edges) total += e.visits;
    double root=std::sqrt(total+1.0), best=-1e300; Edge* chosen=nullptr;
    for(auto& e:n.edges) {
        double q=e.visits ? e.value_sum/e.visits : 0.0;
        double score=q+c*e.prior*root/(1+e.visits);
        if(score>best){best=score;chosen=&e;}
    }
    return chosen;
}
void expand(Node& n, const State& s, const std::array<double,ACTIONS>& policy) {
    double total=0; for(int a=0;a<ACTIONS;++a) if(s.heights[a]<N) total += std::max(0.0,policy[a]);
    int legal=0; for(auto h:s.heights) if(h<N) ++legal;
    for(int a=0;a<ACTIONS;++a) if(s.heights[a]<N) {
        double p=total>0 ? std::max(0.0,policy[a])/total : 1.0/legal;
        n.edges.push_back(Edge{a,p});
    }
}
void noise(Node& n, PyObject* rng, double alpha, double fraction) {
    std::vector<double> samples; double total=0;
    for(size_t i=0;i<n.edges.size();++i){double x=rng_gamma(rng,alpha); samples.push_back(x); total+=x;}
    if(total<=0) return;
    for(size_t i=0;i<n.edges.size();++i) n.edges[i].prior=(1-fraction)*n.edges[i].prior+fraction*samples[i]/total;
}
void backup(const std::vector<Edge*>& path,double value){for(auto it=path.rbegin();it!=path.rend();++it){value=-value;++(*it)->visits;(*it)->value_sum+=value;}}

bool evaluate(PyObject* evaluator, PyObject* state_cls, const std::vector<State>& states,
              std::vector<std::pair<std::array<double,ACTIONS>,double>>& out) {
    PyObject* list=PyList_New((Py_ssize_t)states.size()); if(!list) return false;
    for(Py_ssize_t i=0;i<(Py_ssize_t)states.size();++i){PyObject* s=state_to_python(states[i],state_cls);if(!s){Py_DECREF(list);return false;}PyList_SET_ITEM(list,i,s);}
    PyObject* result=PyObject_CallMethod(evaluator,"evaluate_batch","O",list); Py_DECREF(list); if(!result)return false;
    PyObject* seq=PySequence_Fast(result,"evaluate_batch must return a sequence"); Py_DECREF(result); if(!seq)return false;
    if(PySequence_Fast_GET_SIZE(seq)!=(Py_ssize_t)states.size()){Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"wrong evaluation batch size");return false;}
    out.clear(); out.reserve(states.size());
    for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(seq);++i){
        PyObject* pair=PySequence_Fast(PySequence_Fast_GET_ITEM(seq,i),"evaluation must be (policy, value)"); if(!pair){Py_DECREF(seq);return false;}
        PyObject* pol=PySequence_Fast(PySequence_Fast_GET_ITEM(pair,0),"policy must be a sequence"); if(!pol){Py_DECREF(pair);Py_DECREF(seq);return false;}
        if(PySequence_Fast_GET_SIZE(pol)!=ACTIONS){Py_DECREF(pol);Py_DECREF(pair);Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"policy must have 16 entries");return false;}
        std::array<double,ACTIONS> p{}; for(int a=0;a<ACTIONS;++a)p[a]=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(pol,a));
        double v=PyFloat_AsDouble(PySequence_Fast_GET_ITEM(pair,1)); Py_DECREF(pol);Py_DECREF(pair); if(PyErr_Occurred()){Py_DECREF(seq);return false;}
        out.push_back({p,std::clamp(v,-1.0,1.0)});
    }
    Py_DECREF(seq); return true;
}

std::array<double,ACTIONS> visit_policy(Node& n,double temperature){
    std::array<double,ACTIONS> p{}; int total=0; for(auto& e:n.edges)total+=e.visits;
    if(temperature<=0){auto it=std::max_element(n.edges.begin(),n.edges.end(),[](auto& a,auto& b){return a.visits<b.visits;});p[it->action]=1;return p;}
    double sum=0; for(auto& e:n.edges){p[e.action]=e.visits?std::pow((double)e.visits,1.0/temperature):0;sum+=p[e.action];}
    if(sum<=0){for(auto& e:n.edges)p[e.action]=1.0/n.edges.size();}else for(double& x:p)x/=sum; return p;
}
int sample(const std::array<double,ACTIONS>& p,PyObject* rng){double t=rng_random(rng),c=0;int best=0;for(int a=0;a<ACTIONS;++a){c+=p[a];if(t<=c)return a;if(p[a]>p[best])best=a;}return best;}

PyObject* observation(const State& s){
    PyObject* outer=PyList_New(2);
    for(int ch=0;ch<2;++ch){PyObject* zs=PyList_New(4);for(int z=0;z<4;++z){PyObject* ys=PyList_New(4);for(int y=0;y<4;++y){PyObject* xs=PyList_New(4);for(int x=0;x<4;++x){int8_t want=ch==0?s.to_play:-s.to_play;PyList_SET_ITEM(xs,x,PyFloat_FromDouble(s.board[idx(x,y,z)]==want?1:0));}PyList_SET_ITEM(ys,y,xs);}PyList_SET_ITEM(zs,z,ys);}PyList_SET_ITEM(outer,ch,zs);}return outer;
}
PyObject* build_examples(const Game& g,PyObject* example_cls){
    PyObject* list=PyList_New((Py_ssize_t)g.history.size()); int w=winner(g.state);
    for(Py_ssize_t i=0;i<(Py_ssize_t)g.history.size();++i){auto& h=g.history[i];PyObject* obs=observation(h.state);PyObject* pol=PyList_New(ACTIONS);for(int a=0;a<ACTIONS;++a)PyList_SET_ITEM(pol,a,PyFloat_FromDouble(h.policy[a]));double v=w==0?0:(w==h.state.to_play?1:-1);PyObject* ex=PyObject_CallFunction(example_cls,"OOd",obs,pol,v);Py_DECREF(obs);Py_DECREF(pol);if(!ex){Py_DECREF(list);return nullptr;}PyList_SET_ITEM(list,i,ex);}return list;
}

PyObject* play_games(PyObject*,PyObject* args){
    PyObject *evaluator,*config,*rng; int count; if(!PyArg_ParseTuple(args,"OOiO",&evaluator,&config,&count,&rng))return nullptr;
    auto attr_i=[&](const char* n){PyObject* x=PyObject_GetAttrString(config,n);long v=PyLong_AsLong(x);Py_XDECREF(x);return (int)v;};
    auto attr_d=[&](const char* n){PyObject* x=PyObject_GetAttrString(config,n);double v=PyFloat_AsDouble(x);Py_XDECREF(x);return v;};
    int simulations=attr_i("simulations"), temp_moves=attr_i("temperature_moves"), reuse_tree=attr_i("reuse_tree"); double cpuct=attr_d("c_puct"),alpha=attr_d("dirichlet_alpha"),fraction=attr_d("exploration_fraction");
    PyObject* game_mod=PyImport_ImportModule("score4.game"),*sp_mod=PyImport_ImportModule("score4.self_play");if(!game_mod||!sp_mod){Py_XDECREF(game_mod);Py_XDECREF(sp_mod);return nullptr;}
    PyObject* state_cls=PyObject_GetAttrString(game_mod,"Score4State"),*example_cls=PyObject_GetAttrString(sp_mod,"TrainingExample");Py_DECREF(game_mod);Py_DECREF(sp_mod);if(!state_cls||!example_cls){Py_XDECREF(state_cls);Py_XDECREF(example_cls);return nullptr;}
    std::vector<Game> active(std::max(0,count)), finished;
    while(!active.empty()){
        std::vector<State> roots;std::vector<Game*> root_games;for(auto& g:active)if(g.root->edges.empty()){roots.push_back(g.state);root_games.push_back(&g);}
        std::vector<std::pair<std::array<double,ACTIONS>,double>> ev;if(!roots.empty()&&!evaluate(evaluator,state_cls,roots,ev))goto error;
        for(size_t i=0;i<roots.size();++i)expand(*root_games[i]->root,roots[i],ev[i].first);
        for(auto& g:active)noise(*g.root,rng,alpha,fraction);if(PyErr_Occurred())goto error;
        for(int sim=0;sim<std::max(0,simulations);++sim){
            std::vector<Leaf> leaves;std::vector<State> states;
            for(auto& g:active){State s=g.state;Node* n=g.root.get();std::vector<Edge*> path;while(!terminal(s)&&!n->edges.empty()){Edge* edge=select_edge(*n,cpuct);path.push_back(edge);s=play(s,edge->action);if(!edge->child)edge->child=std::make_unique<Node>();n=edge->child.get();}if(terminal(s))backup(path,terminal_value(s));else{states.push_back(s);leaves.push_back(Leaf{&g,s,n,std::move(path)});}}
            if(!states.empty()&&!evaluate(evaluator,state_cls,states,ev))goto error;
            for(size_t i=0;i<leaves.size();++i){expand(*leaves[i].node,leaves[i].state,ev[i].first);backup(leaves[i].path,ev[i].second);}
        }
        std::vector<Game> next;next.reserve(active.size());for(auto& g:active){auto p=visit_policy(*g.root,g.state.ply<temp_moves?1.0:0.0);int a=sample(p,rng);if(PyErr_Occurred())goto error;std::unique_ptr<Node> child;if(reuse_tree){for(auto& edge:g.root->edges)if(edge.action==a){child=std::move(edge.child);break;}}g.history.push_back(History{g.state,p});g.state=play(g.state,a);g.root=child?std::move(child):std::make_unique<Node>();if(terminal(g.state))finished.push_back(std::move(g));else next.push_back(std::move(g));}active=std::move(next);
    }
    {PyObject* result=PyList_New((Py_ssize_t)finished.size());for(Py_ssize_t i=0;i<(Py_ssize_t)finished.size();++i){PyObject* x=build_examples(finished[i],example_cls);if(!x){Py_DECREF(result);goto error;}PyList_SET_ITEM(result,i,x);}Py_DECREF(state_cls);Py_DECREF(example_cls);return result;}
error: Py_DECREF(state_cls);Py_DECREF(example_cls);return nullptr;
}

PyMethodDef methods[]={{"play_games_batched",play_games,METH_VARARGS,"Run batched self-play in C++."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_self_play_cpp",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__self_play_cpp(){return PyModule_Create(&module);}
