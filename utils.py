import os
import collections
import contextlib
import json
import sys
import wave

import ffmpeg
import webrtcvad

def any2wav(in_filename, out_filename, **input_kwargs):
    """ Convert any audio to PCM with 16k sample rate. """
    try:
        out, err = (ffmpeg
            .input(in_filename, **input_kwargs)
            .output(out_filename, acodec='pcm_s16le', ac=1, ar='16k')
            .global_args('-loglevel', 'error')
            .global_args('-y')
            .run()
        )
    except ffmpeg.Error as e:
        print(e.stderr, file=sys.stderr)
        return False
    return True

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wavfile:
        num_channels = wavfile.getnchannels()
        assert num_channels == 1
        sample_width = wavfile.getsampwidth()
        assert sample_width == 2
        sample_rate = wavfile.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wavfile.readframes(wavfile.getnframes())
        return pcm_data, sample_rate

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)
        wavfile.setframerate(sample_rate)
        wavfile.writeframes(audio)

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

# # Changed output to frames instead of binary data
def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))

                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for tmp_frame, tmp_speech in ring_buffer:
                    voiced_frames.append(tmp_frame)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                # yield b''.join([f.bytes for f in voiced_frames])
                yield voiced_frames
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        pass

    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.

    if voiced_frames:
        # yield b''.join([f.bytes for f in voiced_frames])
        yield voiced_frames

def wav2segments(wavfile, mode=2, outputdir=None, max_duration_ms=20000, is_byte=False, prefix="chunk", min_duration_ms=None):
    """ Convert an wavfile to voiced segments

    Returns:
        list_timestamps: list of dict, {"id": i, "start": start, "stop": stop}
    """
    frame_duration_ms = 30
    padding_duration_ms = 500

    if is_byte:
        audio = wavfile
        sample_rate = 16000
    else:
        audio, sample_rate = read_wave(wavfile)

    vad = webrtcvad.Vad(mode)
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)
    max_number_frame = int(max_duration_ms/frame_duration_ms)
    segments = split_too_long(segments, max_number_frame)
    
    # print(len(segments))

    list_timestamps = []
    for i, voiced_frames in enumerate(segments):
        start = voiced_frames[0].timestamp
        stop = voiced_frames[-1].timestamp + voiced_frames[-1].duration
        timestamp = {
            "id": i, 
            "start": round(start, 4), 
            "stop": round(stop, 4), 
            'length': len(voiced_frames)
        }
        # timestamp = {"id": i, "start": segment.timestamp, "stop": segment.duration}
        list_timestamps.append(timestamp)

        if outputdir is not None and os.path.isdir(outputdir):
            segment = b''.join([f.bytes for f in voiced_frames])
            segment_path = os.path.join(outputdir, "{}-{}.wav".format(prefix, str(i).zfill(4)))
            timestamp['speech'] = segment_path
            write_wave(segment_path, segment, sample_rate)

            with open(os.path.join(outputdir, "timestamps.json"), 'w') as writer:
                json.dump(list_timestamps, writer, indent=4)
    return list_timestamps

def split_too_long(segments, max_number_frame):
    """ Limit the number of frames of each segment.
    Args:
        segments: list of segment, and each segment is a list of frame
        max_number_frame: int
    Returns:
        new_segments: list of segment
    """
    new_segments = []
    for voiced_frames in segments:
        new_voiced_frames = []
        counter = 0
        for frame in voiced_frames:
            if counter == max_number_frame:
                new_segments.append(new_voiced_frames)
                new_voiced_frames = []
                counter = 0
            new_voiced_frames.append(frame)
            counter+=1
        if counter > 0:
            new_segments.append(new_voiced_frames)
    return new_segments
