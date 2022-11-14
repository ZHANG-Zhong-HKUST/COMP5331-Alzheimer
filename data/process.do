clear
insheet using "ROSTER.csv"
keep phase rid ptid
save roster
save merged

clear
insheet using "CDR.csv"
keep phase rid userdate cdrsb
gen date = clock(userdate,"DMY")
gsort -date
duplicates drop phase rid, force
drop date examdate
save cdr

use merged
merge 1:1 phase rid using cdr
drop if _merge<3
drop if missing(cdrsb)
drop _merge

save, replace

clear
insheet using "ADAS_ADNIGO23.csv"
keep phase rid total13 userdate userdate2
gen date = clock(userdate,"DMY")
gsort -date
duplicates drop phase rid, force
drop userdate userdate2 date
rename total13 ADASSCORE
save ADAS13

clear
insheet using "ADASSCORES.csv"
gen date = clock(userdate,"DMY")
gsort -date
duplicates drop rid, force
gen phase = "ADNI1"
keep phase rid total11
rename total11 ADASSCORE
append using ADAS13
save ADAS

use merged
merge 1:1 phase rid using ADAS
drop if _merge<3
drop if missing(ADASSCORE)
drop _merge
save, replace

clear
insheet using "MMSE.csv"
keep phase rid total13 userdate userdate2
gen date = clock(userdate,"DMY")
gsort -date
duplicates drop phase rid, force
drop userdate date
save mmse

use merged
merge 1:1 phase rid using mmse
drop if _merge<3
drop if missing(mmscore)
drop _merge
save, replace

clear
insheet using "FAQ.csv"
keep phase rid faqtotal userdate
gen date = clock(userdate,"DMY")
gsort -date
duplicates drop phase rid, force
drop userdate date
save faq

use merged
merge 1:1 phase rid using faq
drop if _merge<3
drop if missing(faqtotal)
drop _merge
save, replace

clear
insheet using "WHOLEBRAIN.csv"
rename colprot phase
drop viscode viscode2 manufacturer manufacturersm
drop status update_stamp
drop magneticfieldstrength mracquisitiontypeflair
gen date = clock(examdate,"DMY")
gsort -date
duplicates drop phase rid, force
drop date examdate

use merged
merge 1:1 phase rid using wholebrain
drop if _merge<3
drop _merge
save, replace

clear
insheet using "NEUROBAT.csv"
drop id siteid viscode viscode2 userdate2 examdate
gen date = clock(userdate,"DMY")
gsort -date
duplicates drop phase rid, force
drop userdate date
save RAVLT


clear
insheet using "APOERES.csv"
drop id siteid viscode userdate2 update_stamp aptestdt
gen date = clock(userdate,"DMY")
gsort -date
duplicates drop phase rid, force
drop userdate date
save APOE

use merged
merge 1:1 phase rid using APOE
drop if _merge<3
drop _merge
drop apvolume apreceive apambtemp apresamp apusable
save, replace


clear
insheet using "UCBERKELEYFDG.csv"
rename colprot phase
gen date = clock(examdate,"DMY")
gsort -date
duplicates drop phase rid roiname, force
drop viscode viscode2 examdate uid date update_stamp
egen roinameid = group(roiname)
drop roiname
reshape wide mean max stdev totvox ,i(phase rid) j(roinameid)
save FDG

use merged
merge 1:1 phase rid using faq
drop if _merge<3
drop _merge
save merged_2, replace