#ifndef DLLFILEAUX_HPP
#define DLLFILEAUX_HPP
/** \file
 * Use dllbuild.hpp to create a 'make' subdirectory, create lib, read symbols.
 */
#include "vednnx.h"
#include "conv_test_param.h"    // param_cstr etc.
#include "cblock.hpp"           // Cblock and CSTR macro
#include "stringutil.hpp"
#include <algorithm>    // unique

/** Modify \c str squishing repeated \c squish [space] to single \c squish. */
inline std::string squishRepeats(std::string str, char squish=' '){
    std::string::iterator new_end =
        std::unique(str.begin(), str.end(),
                [&squish](char lhs, char rhs){return lhs==squish && rhs==squish;}
                );
    str.erase(new_end, str.end());
    return str;
}
/** Adjust \c str removing all \c squish chars after a \c after char [Default: all blanks after a newline]. */
inline std::string squishAfter(std::string str, char const* after="\n", char const* squish=" "){
    std::string ret;
    ret.reserve(str.size());
    size_t ok=0, aft, ok2;
    while((aft=str.find_first_of(after,ok)) != std::string::npos){
        ret.append(str,ok,aft-ok+1);
        //cout<<" [ok,aft]=["<<ok<<","<<aft<<"] ret="<<ret<<endl;
        if((ok2=str.find_first_not_of(squish,aft+1))!=std::string::npos){
            ok=ok2;
        }else{
            break;
        }
    }
    //cout<<" str["<<str.size()<<"] ok="<<ok<<" aft="<<aft<<endl;
    if(ok<str.size()) ret.append(str.substr(ok));
    return ret;
}
/** Align all \c delim lines prepending spaces before the first \c match string so they are aligned.
 * Note: you should \c match beginning-of-word to avoid inserting space inside a word. */
inline std::string alignRight(std::string match, std::string str, char const* delim="\n"){
    std::vector<size_t> m; // match pos
    std::vector<size_t> l; // prev line start
    size_t loc=0, mat;
    size_t col=0;
    //cout<<" alignRight( str["<<str.size()<<"], match=\""<<match<<"\", delim)"<<endl;
    while((mat=str.find(match,loc)) != std::string::npos){
        m.push_back(mat);
        size_t d = str.find_last_of(delim, mat);
        if(d==std::string::npos) d=0;
        l.push_back(d);                         // m-l is column to be aligned
        col = std::max(col, mat - d);
        //cout<<" loc="<<loc<<" --> match[d,mat]=["<<d<<","<<mat<<"] col="<<col<<endl;
        //cout<<" line = "<<str.substr(d,mat-d)<<endl;
        loc = str.find_first_of(delim,mat);   // skip to next line 'loc'
        //cout<<" loc' = "<<loc<<endl;
    }
    std::string ret;
    size_t insert=0;
    for(size_t i=0; i<m.size(); ++i){
        insert += col - (m[i] - l[i]);
    }
    if(insert==0){
        ret = str;
    }else{
        //cout<<" original str: "<<str<<endl;
        ret.reserve(str.size()+insert);
        size_t at = 0;
        for(size_t i=0; i<m.size(); ++i){
            ret.append(str, at, m[i]-at);
            insert = col - (m[i] - l[i]);
            ret.append(insert, ' ');
            //cout<<" at="<<at<<" l[i],m[i]="<<l[i]<<","<<m[i]<<" ret={"<<ret<<"}"<<endl;
            at = m[i];
        }
        if(at<str.size()){
            //cout<<" final at="<<at<<" m.back()="<<m.back()<<"  add final "<<str.substr(at)<<endl;
            ret.append(str.substr(at));
        }
    }
    return ret;
}
/** This ends up in symbols names quite frequently */
inline std::string paramString(struct param const* const p){
    int const sz=512;
    char buf[sz];
    // param_cstr exits on error, returning &buf[0] null-terminated C-string
    //return std::string(param_cstr(p,&buf[0],sz-1));
    //  Eventually I tired of the long symbol names...
    // NEW: output 0-based dilation (mkl-dnn convention)
    // NEW: shorten Height==Width common cases
    return std::string(param_cstr_short(p,&buf[0],sz-1));
}

/** additional useful information about a libvednn layer, related
 * to naming conventions. */
struct DllFileAux {
    // all strings empty if layer+dirn is not recognized
    DllFileAux(std::string layer, std::string dirn);
    std::string layer;          // ex. Convolution
    std::string dirn;           // ex. Forward, ForwardAddBias, BackwardData, etc.
    std::string layer_mac;      // ex. CONV
    std::string dirn_mac;       // ex. FORWARDADDBIAS, BACKWARD_DATA
    std::string dirn_mac_short; // ex. FWD, FWDB, BKWD, BKWF
    std::string layer_short;    // ex. Conv
    std::string dirn_short;     // ex. Fwd, Fwdb, Bkwd, Bkwf ??
    std::string layer_type;     // ex. ConvForward, or maybe layer_short + dirn_short ??

    std::string impl_decl_args;
    std::string impl_ok_decl_args;
    std::string impl_rtok_decl_args;
    std::string impl_Begin_decl_args;
    std::string impl_Next_decl_args;
    std::string impl_realNext_decl_args;
    std::string impl_Dump_decl_args;
    std::string impl_Run_decl_args;

    std::string impl_declare(std::string basename);
    std::string impl_ok_declare(std::string basename);  // vednnError_t basename_ok(...)
    std::string impl_rtok_declare(std::string basename);
    std::string impl_Begin_declare(std::string basename);
    std::string impl_Next_declare(std::string basename);
    std::string impl_realNext_declare(std::string basename);
    std::string impl_Dump_declare(std::string basename);
    std::string impl_Run_declare(std::string basename);
};

inline DllFileAux::DllFileAux(std::string arg_layer, std::string arg_dirn)
{
    if(arg_layer == "Convolution"){
        if(arg_dirn == "Forward"){
            layer="Convolution";
            dirn="Forward";
            layer_mac="CONV";
            dirn_mac="FORWARD";
            dirn_mac_short="FWD";
            layer_short="Conv";
            dirn_short="Fwd";
            layer_type="vednn"+layer_short+dirn;

            // This is actually a macro-friendly template for all layers ...
            //std::string p(CSTR(VEDNN_PARAMS_CONV_FORWARD));
            //std::string d(CSTR(VEDNN_DATARG_CONV_FORWARD));
            impl_decl_args=CSTR(CONVX_FWD_ORDER(VEDNN_PARAMS_CONV_FORWARD,VEDNN_DATARG_CONV_FORWARD));
            impl_ok_decl_args=CSTR(VEDNN_PARAMS_CONV_FORWARD);
            impl_rtok_decl_args=CSTR(VEDNN_DATARG_CONV_FORWARD);
            impl_Begin_decl_args=CSTR(VEDNN_PARAMS_CONV_FORWARD);
            impl_Next_decl_args=layer_type+"Impls* current," CSTR(VEDNN_PARAMS_CONV_FORWARD);
            impl_realNext_decl_args=layer_type+"Impls* current,"
                CSTR(VEDNN_PARAMS_CONV_FORWARD,VEDNN_DATARG_CONV_FORWARD);
            impl_Dump_decl_args=layer_type+"Impls const* current";
            impl_Run_decl_args=CSTR(VEDNN_PARAMS_CONV_FORWARD,VEDNN_DATARG_CONV_FORWARD);
        }
    }
}
inline std::string DllFileAux::impl_declare(std::string basename){
    return alignRight("*",
            multiReplace(",",",\n    ",
                "vednnError_t "+basename+"(\n    "
                +squishAfter(impl_decl_args, ",", " \t")
                +"\n    )"
                ));
}
inline std::string DllFileAux::impl_ok_declare(std::string basename){
    return alignRight("*",multiReplace(",",",\n    ",
                "vednnError_t "+basename+"_ok(\n    "
                +squishAfter(impl_ok_decl_args, ",", " \t")
                +"\n    )"
                ));
}
inline std::string DllFileAux::impl_rtok_declare(std::string basename){
    return alignRight("*",multiReplace(",",",\n    ",
                "vednnError_t "+basename+"_rtok(\n    "
                +squishAfter(impl_rtok_decl_args, ",",  " \t")
                +"\n    )"
                ));
}
inline std::string DllFileAux::impl_Begin_declare(std::string basename){
    return alignRight("*",multiReplace(",",",\n    ",
                "vednnError_t "+basename+"_Begin(\n    "
                +squishAfter(impl_Begin_decl_args, ",",  " \t")
                +"\n    )"
                ));
}
inline std::string DllFileAux::impl_Next_declare(std::string basename){
    //std::string layer_type("vednnConvForward");
    return alignRight("*",multiReplace(",",",\n    ",
                "vednnError_t "+basename+"_Next(\n    "
                +squishAfter(impl_Next_decl_args, ",",  " \t")
                +"\n    )"
                ));
}
inline std::string DllFileAux::impl_realNext_declare(std::string basename){
    return alignRight("*",multiReplace(",",",\n    ",
                "vednnError_t "+basename+"_realNext(\n    "
                +squishAfter(impl_realNext_decl_args, ",",  " \t")
                +"\n    )"
                ));
}
inline std::string DllFileAux::impl_Dump_declare(std::string basename){
    return alignRight("*",multiReplace(",",",\n    ",
                "vednnError_t "+basename+"_Dump(\n    "
                +squishAfter(impl_Dump_decl_args, ",",  " \t")
                +"\n    )"
                ));
}
inline std::string DllFileAux::impl_Run_declare(std::string basename){
    return alignRight("*",multiReplace(",",",\n    ",
                layer_type+"_out_t "+basename+"_Run(\n    "
                +squishAfter(impl_Run_decl_args, ",",  " \t")
                +"\n    )"
                ));
}
// vim: ts=4 sw=4 et cindent cino=^=l0,\:.5s,=-.5s,N-s,g.5s,b1 cinkeys=0{,0},0),\:,0#,!^F,o,O,e,0=break
#endif // DLLFILEAUX_HPP
