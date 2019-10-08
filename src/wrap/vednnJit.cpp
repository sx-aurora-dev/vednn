#include "vednnJit.h"
#include "vednnJitDetail.hpp"   // VednnJitCtx (symbol resolution, libraries)
#include "vednnJitDetail.h"     // Symbol internals
#include <assert.h>
#include <string.h>
#include <dlfcn.h>
#include <stdio.h>

using namespace vednn::wrap;

extern "C" { //}

vednnError_t vednnInitJit(void** jitctx,
        char const* const process_subdir,
        char const* const process_lib){
    VednnJitCtx *ctx = new VednnJitCtx(process_subdir,process_lib);
    *jitctx = (void*)ctx;
    return (*jitctx? VEDNN_SUCCESS: VEDNN_ERROR_MEMORY_EXHAUST);
}

vednnError_t vednnFreeJit(void** jitctx){
    delete (VednnJitCtx*)(*jitctx);
    *jitctx = nullptr;
    return VEDNN_SUCCESS;
}

vednnError_t vednnJitLib(void* jitctx, char const* libpath){
    int err=0;
    if(libpath && jitctx){
        ((VednnJitCtx*)jitctx)->addlib(libpath); // add error ret?
        return VEDNN_SUCCESS;
    }else{
        err=1;
    }
    return (err==0? VEDNN_SUCCESS: VEDNN_ERROR_INVALID_PARAM);
}

/** There is \b no usage counter to update \c VednnJitCtx :(.
 * [paranoia] We take a ptr-to-ptr and NULL everything. */
void vednnSym_free( vednnSym_t ** jitsym ){
    // construct utility wrapper around *jitsym (exposing opaque info nicely)
    Sym(**jitsym).erase(); // zero things, set ptrs to NULL, free memory.
    delete *jitsym;
    // Finally make the pointer itself NULL
    jitsym = nullptr;
}

}//"C"


#define JITCTX_PATHMAX 1024

char const* VednnJitCtx::dup(char const* const src){
    char * dst = nullptr;
    if(src){
        auto len = strnlen(src, JITCTX_PATHMAX+1);
        char * dst = new char[len+1];
        if(dst) strncpy( dst, src, JITCTX_PATHMAX);
    }
    return (char const*)dst;
}

VednnJitCtx::VednnJitCtx(
        char const* const process_subdir,
        char const* const process_libpath )
    : seq(0),
    process_subdir(nullptr),
    process_libpath(nullptr),
    root({nullptr,nullptr,0,nullptr})
{
    if(process_subdir){
        this->process_subdir = this->dup(process_subdir);
    }
    if(process_libpath){
        this->process_libpath = this->dup(process_libpath);
    }
    dlerror();
    root.dlHandle = dlopen(root.libPath, // nullptr is the correct value
            RTLD_LAZY );
    char const* err = dlerror();
    root.seq = (err? 0U: ++this->seq);
    //
    if(this->process_libpath){
        // constructor guarantee: root->next exists too
        // It is ok for its dlHandle to be NULL
        if(!root.next){
            root.next = newLibs(this->process_libpath);
        }
    }
    mkseqs();
}

VednnJitCtx::~VednnJitCtx()
{
    Libs* nxt = &root;
    while(nxt){
        Libs *cur = nxt;
        nxt = cur->next;    // remember next item
        this->del(cur);     // completely release cur
    }
    nseqs = 0U;
    delete[] process_subdir;
    delete[] process_libpath;
}

void VednnJitCtx::addlib(char const* libPath){
    // check if libPath matches existing list entry, and if possible,
    // mark as stale (or warn)
    //int err = 1;
    Libs* tail = &root;
    Libs* prev = nullptr;
    for(;tail;){
        bool tailUnlinked = false;
        // check for match as tail iterates toward end of list
        if( tail->libPath && strncmp(libPath,tail->libPath,JITCTX_PATHMAX)==0 ){
            reopen(tail); // reopen, mark stale
            if(tail->dlHandle){
                //err = 0;
                break; // EXIT !
            }else{ // repoen failed, for some reason
                // NO. else leave entry there, syms may need to check staleness.
                // YES. with contains(sq) have easier way to check, and can get
                //      rid of entries.
                if(prev){
                    prev->next = tail->next;
                    tailUnlinked = true;
                }
            }
        }
        // check for tail REALLY at end list
        if(!tail->next){
            // add libPath as a completely new list entry.
            Libs *nxt = newLibs(libPath);
            if(nxt->dlHandle){ // everything seems fine, add entry!
                tail->next = nxt; //err = 0;
            }else{
                this->del(nxt); // unused, get rid of  it
            }
            break; // EXIT !
        }
        prev = tail;
        tail = tail->next; // no match! next Libs entry
        if(tailUnlinked){
            del(prev);
        }
    }
    mkseqs(); // Paranoia!
}

bool VednnJitCtx::mkseqs(){
    bool ok = true;
    nseqs = 0U;
    Libs* nxt = &root;
    while(nxt){
        Libs *cur = nxt;
        nxt = cur->next;    // remember next item
        if(cur->seq){
            if(nseqs >= VEDNNJIT_MAXLIBS){
                ok = false; // 
                printf(" Warning: VednnJitCtx exceeded max libraries %llu\n"
                        "          Consider combining some .a and converting to a large .so\n"
                        , (unsigned long long)VEDNNJIT_MAXLIBS );
                break;
            }
            seqs[nseqs] = cur->seq;
            ++nseqs;
        }
    }
    return ok;
}
VednnJitCtx::Libs * VednnJitCtx::newLibs(char const* libPath){
    Libs *nxt = new Libs({nullptr,nullptr,0,nullptr});
    nxt->libPath = this->dup(libPath);
    dlerror();
    nxt->dlHandle = dlopen(nxt->libPath,
            RTLD_LAZY );
    char const *err = dlerror();
    nxt->seq = (err? 0U: ++this->seq);
    if(err) printf(" Warning: VednnJitCtx::newLibs(%s) : %s\n",
            (libPath? libPath: "NULL"), err );
    return nxt;
}

void VednnJitCtx::del( Libs* old ){ // full removal
    // Don't care about the forward linked list from root.
    // Behave as though old has been properly unlinked from the forward list.
    delete[] old->libPath;
    old->libPath=nullptr;
    dlerror();
    dlclose(old->dlHandle);
    char const* err = dlerror();
    if(err) printf(" Warning : VednnJitCtx::del(%s) : %s\n",
            (old->libPath? old->libPath: "NULL"), err);
    old->dlHandle = nullptr;
    old->seq = 0;
    old->next = nullptr;
    delete old;
}

void VednnJitCtx::reopen( Libs* stale ){
    if(stale){
        char const* err;
        int nerr = 0;
        dlerror();
        dlclose(stale->dlHandle);
        err = dlerror();
        if(err){
            //++nerr; ignore
            printf(" dlclose error during VednnJitCtx::reopen dlclose of %s : %s\n",
                    (stale->libPath? stale->libPath: "NULL"), err);
        }
        stale->dlHandle = nullptr;
        stale->dlHandle = dlopen(stale->libPath,
                RTLD_LAZY );
        err = dlerror();
        if(err){
            ++nerr;
            printf(" dlopen error during VednnJitCtx::reopen dlopen of %s : %s\n",
                    (stale->libPath? stale->libPath: "NULL"), err);
            assert(stale->dlHandle == nullptr);
        }
        stale->seq = (nerr? 0U: ++this->seq); // tentative assignment
        if(!mkseqs()){ // ohoh too many open libraries? fatal error?
            ++nerr;
            stale->seq = 0U;
            mkseqs();
        }
    }
}

// vim: et ts=4 sw=4 cindent cino=l1,)0,u0,W2,\:0,=2s,N-s,g-2,h2 syntax=cpp.doxygen
