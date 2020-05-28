#include "config.h"

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <mutex>

#include "Utils.h"
#include "cfg.h"

static std::mutex IOmutex;

void Utils::auto_printf(const char *fmt, ...) {
    if (cfg_quiet) return;

    va_list ap;
    va_start(ap, fmt);
	if (cfg_logfile_stream) {
		std::lock_guard<std::mutex> lock(IOmutex);
		vfprintf(cfg_logfile_stream, fmt, ap);
	} else {
    	vfprintf(stdout, fmt, ap);
	}
    va_end(ap);
}

void Utils::stream_printf(const char *fmt, ...) {
	va_list ap;
    va_start(ap, fmt);
	if (cfg_logfile_stream) {
		std::lock_guard<std::mutex> lock(IOmutex);
		vfprintf(cfg_logfile_stream, fmt, ap);
	} else {
    	vfprintf(stdout, fmt, ap);
	}
    va_end(ap);
}

void Utils::gtp_printf(const char *fmt, ...) {
	//if (!cfg_gtp_mode) return; 

	va_list ap;
    va_start(ap, fmt);
	fprintf(stdout, "= ");
	vfprintf(stdout, fmt, ap);
	fprintf(stdout, "\n");
	va_end(ap);
}

void Utils::gtp_fail_printf(const char *fmt, ...) {
	va_list ap;
    va_start(ap, fmt);
	fprintf(stdout, "? ");
	vfprintf(stdout, fmt, ap);
	fprintf(stdout, "\n");
	va_end(ap);
}
