# Platypus: Lab book
This file contains lab notes on the **Platypus** project.
It is prepend-only.
Entry titles should be the date the post is *written*,
not the date of execution of the work it describes.

## 2016-01-13
- DWH did major refactoring, all still works.
- It appears that K=1024 is too far -- clusters break up
- So we should really "just" do 256 and 512

## 2016-01-12
- DWH spoke to DFM yesterday about clustering methods.
DFM strongly advised k-means.
- DWH implemented k-means and it really looks like it is going to work.
- DWH tentatively added DFM to the author list.

## 2016-01-11
- DWH found abundance-space structure on 2016-01-09 at the Cocoa Beach *APOGEE* meeting.
- On 2016-01-10 he implemented friends-of-friends clustering.
This is not a good idea:
We should be doing some kind of density estimation or else just choosing clusters by hand.
- Today, DWH proposed to HWR, AC, MKN that we just find clusters in abundance space,
and then look at them in configuration space.
