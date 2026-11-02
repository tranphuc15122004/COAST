from problems import VRPTW_Environment
import torch


class DVRPTW_Environment(VRPTW_Environment):
    """Event-driven DVRPTW environment.

    ``cust_mask`` stores only unrevealed requests and padded nodes.  ``mask`` is
    rebuilt from that state, served requests, per-vehicle capacity, and vehicle
    completion. This avoids XORing a reveal bit through a mask that also holds
    unrelated feasibility constraints.
    """

    CUST_FEAT_SIZE = 7

    def _padding_mask(self):
        if self.init_cust_mask is None:
            return torch.zeros_like(self.cust_mask)
        return self.init_cust_mask

    def _refresh_mask(self):
        overload = (
            self.vehicles[:, :, 2:3] - self.nodes[:, None, :, 2] < 0
        )
        self.mask = (
            self.cust_mask[:, None, :]
            | self.served[:, None, :]
            | overload
            | self.veh_done[:, :, None]
        )
        self.mask[:, :, 0] = False

    def _update_mask(self, cust_idx):
        self.new_customers = False
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        self._refresh_mask()

    def _update_hidden(self):
        self.new_customers = False
        real_hidden = self.cust_mask & ~self._padding_mask()
        current_time = self.cur_veh[:, 0, 3]
        reveal = real_hidden & (self.nodes[:, :, 6] <= current_time[:, None])
        if not reveal.any():
            return

        self.cust_mask = self.cust_mask & ~reveal
        self.new_customers = True
        reveal_rows = reveal.any(dim=1) & ~self._finished
        if reveal_rows.any():
            # A vehicle that returned to the depot earlier is available again
            # when a new request arrives. Keep its event clock at least at the
            # reveal time so it cannot act before the newly observed event.
            self.vehicles[reveal_rows, :, 3] = torch.maximum(
                self.vehicles[reveal_rows, :, 3],
                current_time[reveal_rows, None],
            )
            self.veh_done[reveal_rows] = False
        self._refresh_mask()
        self._update_cur_veh()

    def _advance_finished_fleets_to_reveal(self):
        """Wait only after every vehicle in an episode has returned to depot."""
        real_hidden = self.cust_mask & ~self._padding_mask()
        all_vehicles_done = self.veh_done.all(dim=1)
        waiting_rows = all_vehicles_done & real_hidden.any(dim=1) & ~self._finished
        if not waiting_rows.any():
            return

        reveal_times = self.nodes[:, :, 6].masked_fill(~real_hidden, float("inf"))
        next_reveal = reveal_times.min(dim=1).values
        self.vehicles[waiting_rows, :, 3] = torch.maximum(
            self.vehicles[waiting_rows, :, 3],
            next_reveal[waiting_rows, None],
        )
        self.veh_done[waiting_rows] = False
        self._refresh_mask()
        self._update_cur_veh()

    def reset(self):
        self.vehicles = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE)
        )
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]
        self.vehicles[:, :, 2] = self.veh_capa

        self.veh_done = self.nodes.new_zeros(
            (self.minibatch_size, self.veh_count), dtype=torch.bool
        )
        self._finished = self.nodes.new_zeros(
            (self.minibatch_size,), dtype=torch.bool
        )
        self.done = False

        self.cust_mask = self.nodes[:, :, 6] > 0
        if self.init_cust_mask is not None:
            self.cust_mask = self.cust_mask | self.init_cust_mask
        self.new_customers = True
        self.served = torch.zeros_like(self.cust_mask)

        self.cur_veh_idx = self.nodes.new_zeros(
            (self.minibatch_size, 1), dtype=torch.int64
        )
        self.cur_veh = self.vehicles.gather(
            1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
        )
        self._refresh_mask()
        self.cur_veh_mask = self.mask.gather(
            1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count),
        )

    def step(self, cust_idx):
        """Apply one action and charge pending cost only at true termination."""
        active_rows = ~self._finished
        dest = self.nodes.gather(
            1,
            cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE),
        )
        dist, late = self._update_vehicles(dest)
        reward = -dist - self.late_cost * late
        reward[~active_rows] = 0

        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()

        self._advance_finished_fleets_to_reveal()
        self._update_hidden()

        real_customers = ~self._padding_mask()
        real_customers[:, 0] = False
        terminal_rows = (
            self.veh_done.all(dim=1)
            & ~((self.cust_mask & ~self._padding_mask()).any(dim=1))
            & ~self._finished
        )
        if terminal_rows.any():
            pending = (real_customers & ~self.served).sum(dim=1, keepdim=True)
            reward[terminal_rows] -= (
                self.pending_cost * pending[terminal_rows].to(reward.dtype)
            )
            self._finished |= terminal_rows

        self.done = bool(self._finished.all())
        return reward
